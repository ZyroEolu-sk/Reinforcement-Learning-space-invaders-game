import argparse
import os
import shutil
from pathlib import Path
from typing import Callable
import numpy as np
import math
from scipy import stats
import torch

from gym_env_vision import SpaceInvadersVisionEnv
from custom_cnn import SpaceInvadersResidualSiluCNN

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_OBJECTS = {
    "features_extractor_class": SpaceInvadersResidualSiluCNN,
}

class ClearRateCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.completed_episodes = 0
        self.cleared_episodes = 0
        self.curriculum_completed_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                self.completed_episodes += 1
                if info.get("completed_game", False):
                    self.cleared_episodes += 1
                if info.get("curriculum_completed", False):
                    self.curriculum_completed_episodes += 1

        if self.completed_episodes > 0:
            clear_rate = self.cleared_episodes / self.completed_episodes
            curriculum_clear_rate = self.curriculum_completed_episodes / self.completed_episodes
            self.logger.record("rollout/clear_rate", clear_rate)
            self.logger.record("rollout/curriculum_clear_rate", curriculum_clear_rate)

        return True


def make_env_fn(args) -> Callable[[], Monitor]:
    def _factory():
        return Monitor(
            SpaceInvadersVisionEnv(
                render_mode=None,
                max_steps=args.max_steps,
                frame_skip=args.frame_skip,
                start_level=args.start_level,
                max_level=args.max_level,
                img_width=args.img_width,
                img_height=args.img_height,
                enable_combo_actions=args.enable_combo_actions,
            )
        )
    return _factory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training from an existing vision DQN model for Space Invaders."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/vision/best_model/best_model",
        help="Path to an existing DQN model (.zip optional).",
    )
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        default=3000000,
        help="Extra timesteps to train from the loaded model.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/vision",
        help="Directory where continued-training outputs will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel training environments (2-4 óptimo para DQN en M2).")
    parser.add_argument("--frame-skip", type=int, default=2, help="Frames to skip per action.")
    parser.add_argument("--max-steps", type=int, default=24000, help="Max steps per episode.")
    parser.add_argument("--img-width", type=int, default=None, help="Observation frame width after preprocessing (auto from model if omitted).")
    parser.add_argument("--img-height", type=int, default=None, help="Observation frame height after preprocessing (auto from model if omitted).")
    parser.add_argument("--start-level", type=int, default=1, help="Curriculum initial level (1-4).")
    parser.add_argument("--max-level", type=int, default=4, help="Curriculum target level (1-4).")
    parser.add_argument("--eval-freq", type=int, default=25000, help="Evaluation frequency in timesteps.")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Episodes per periodic evaluation (lower is faster).")
    parser.add_argument("--comparison-episodes", type=int, default=50, help="Episodes used in final model comparison.")
    parser.add_argument("--checkpoint-freq", type=int, default=25000, help="Checkpoint frequency in timesteps.")
    
    # Hiperparámetros adaptados a la naturaleza de DQN
    parser.add_argument(
        "--override-learning-rate",
        type=float,
        default=1e-4,
        help="Override learning rate for continued training.",
    )
    parser.add_argument(
        "--override-exploration-initial-eps",
        type=float,
        default=0.20,
        help="Forzar un épsilon inicial al reanudar (ej. 0.20 significa 20%% de acciones aleatorias para romper estancamientos).",
    )
    parser.add_argument(
        "--override-exploration-final-eps",
        type=float,
        default=0.02,
        help="Épsilon mínimo residual al acabar la fase de exploración (ej. 0.02).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",  # DQN suele responder mejor a ritmos de aprendizaje constantes
        choices=["constant", "linear", "cosine"],
        help="Learning-rate schedule to use when continuing training.",
    )
    return parser.parse_args()


def _resolve_project_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _resolve_model_path(path: str) -> str:
    path = _resolve_project_path(path)
    if os.path.isfile(path):
        return path
    zip_path = f"{path}.zip"
    if os.path.isfile(zip_path):
        return zip_path
    raise FileNotFoundError(f"No se encontró el modelo en '{path}' ni en '{zip_path}'.")


def _load_dqn(path: str, **kwargs) -> DQN:
    return DQN.load(path, custom_objects=CUSTOM_OBJECTS, **kwargs)


def _assert_residual_extractor(model: DQN, model_path: str) -> None:
    extractor_name = type(model.policy.features_extractor).__name__
    if extractor_name != "SpaceInvadersResidualSiluCNN":
        raise ValueError(
            "Este script está forzado al extractor personalizado SpaceInvadersResidualSiluCNN. "
            f"El modelo cargado desde '{model_path}' usa '{extractor_name}'."
        )
    print(f"[model] extractor={extractor_name}")

def _infer_image_size_from_model(model: DQN) -> tuple[int, int]:
    shape = getattr(model.observation_space, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError(f"Unsupported observation shape in model: {shape}")

    if shape[0] == 4:
        return int(shape[2]), int(shape[1])
    if shape[2] == 4:
        return int(shape[1]), int(shape[0])

    raise ValueError(f"Cannot infer image size from observation shape: {shape}")


def _infer_combo_actions_from_model(model: DQN) -> bool:
    n_actions = int(getattr(model.action_space, "n", 0))
    if n_actions == 6:
        return True
    if n_actions == 4:
        return False
    raise ValueError(f"Unsupported action-space size in model: {n_actions}")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    return lambda progress_remaining: progress_remaining * initial_value


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    return lambda progress_remaining: 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining))) * initial_value


def _apply_hyperparameter_overrides(model: DQN, args) -> None:
    # Lógica de overrides adaptada a DQN
    lr = float(args.override_learning_rate) if args.override_learning_rate is not None else 1e-4
    if getattr(args, "lr_schedule", "constant") == "constant":
        model.learning_rate = lr
        model.lr_schedule = get_schedule_fn(lr)
    elif args.lr_schedule == "linear":
        model.learning_rate = lr
        model.lr_schedule = linear_schedule(lr)
    elif args.lr_schedule == "cosine":
        model.learning_rate = lr
        model.lr_schedule = cosine_schedule(lr)

    print(f"[override] learning_rate={lr} schedule={args.lr_schedule}")

    # Forzar nuevos rangos de exploración si hay estancamiento (anti-plateau de DQN)
    if args.override_exploration_initial_eps is not None:
        model.exploration_initial_eps = float(args.override_exploration_initial_eps)
        print(f"[override] exploration_initial_eps={model.exploration_initial_eps}")
    if args.override_exploration_final_eps is not None:
        model.exploration_final_eps = float(args.override_exploration_final_eps)
        print(f"[override] exploration_final_eps={model.exploration_final_eps}")


def main():
    args = parse_args()
    print("[forced-custom-cnn] Continue mode requires SpaceInvadersResidualSiluCNN checkpoints.")

    model_path = _resolve_model_path(args.model_path)
    model_probe = _load_dqn(model_path)
    _assert_residual_extractor(model_probe, model_path)
    inferred_width, inferred_height = _infer_image_size_from_model(model_probe)
    args.enable_combo_actions = _infer_combo_actions_from_model(model_probe)
    args.img_width = inferred_width if args.img_width is None else int(args.img_width)
    args.img_height = inferred_height if args.img_height is None else int(args.img_height)

    save_dir = _resolve_project_path(args.save_dir)

    os.makedirs(save_dir, exist_ok=True)
    checkpoints_dir = os.path.join(save_dir, "checkpoints_continued")
    logs_dir = os.path.join(save_dir, "logs_continued")
    final_best_model_dir = os.path.dirname(model_path)
    continued_best_model_dir = os.path.join(save_dir, "best_model_continued")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(final_best_model_dir, exist_ok=True)
    os.makedirs(continued_best_model_dir, exist_ok=True)

    num_envs = max(1, args.num_envs)
    train_factories = [make_env_fn(args) for _ in range(num_envs)]
    train_env_base = SubprocVecEnv(train_factories) if num_envs > 1 else DummyVecEnv(train_factories)
    eval_env_base = DummyVecEnv([make_env_fn(args)])
    train_env = VecTransposeImage(train_env_base)
    eval_env = VecTransposeImage(eval_env_base)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="dqn_space_invaders_vision_continued", 
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=continued_best_model_dir,
        log_path=logs_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=max(1, int(args.n_eval_episodes)),
        deterministic=True,
        render=False,
    )
    clear_rate_callback = ClearRateCallback()

    # Forzar la carga en el acelerador M2 de Apple "mps"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = _load_dqn(
        model_path,
        env=train_env,
        tensorboard_log=logs_dir,
        seed=args.seed,
        device=device
    )
    _assert_residual_extractor(model, model_path)
    _apply_hyperparameter_overrides(model, args)
    
    print(f"[config] num_envs={num_envs} device={device} batch_size={model.batch_size}")

    interrupted_snapshot_path = os.path.join(continued_best_model_dir, "interrupted_last_model")
    interrupted_snapshot_zip = f"{interrupted_snapshot_path}.zip"

    try:
        model.learn(
            total_timesteps=args.additional_timesteps,
            callback=[checkpoint_callback, eval_callback, clear_rate_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[interrupted] Entrenamiento interrumpido por usuario. Guardando snapshot para comparar...")
        model.save(interrupted_snapshot_path)

    final_zip_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
    eval_best_model_path = os.path.join(continued_best_model_dir, "best_model.zip")

    candidate_paths = []
    if os.path.isfile(eval_best_model_path):
        candidate_paths.append(("eval_best", eval_best_model_path))
    if os.path.isfile(interrupted_snapshot_zip):
        candidate_paths.append(("interrupted_snapshot", interrupted_snapshot_zip))

    if candidate_paths and os.path.isfile(final_zip_path):
        print("\n[model-comparison] Comparación estadística (t-test) del modelo actual vs candidatos...")
        try:
            comparison_episodes = max(1, int(args.comparison_episodes))
            current_best = _load_dqn(final_zip_path) # Cargar con la clase DQN

            current_returns = []
            for i in range(comparison_episodes):
                obs = eval_env.reset()
                done = False
                episode_return = 0.0
                while not done:
                    action, _ = current_best.predict(obs, deterministic=True)
                    obs, rewards, dones, _ = eval_env.step(action)
                    episode_return += float(rewards[0])
                    done = bool(dones[0])
                current_returns.append(episode_return)
            
            current_mean = float(np.mean(current_returns))
            current_std = float(np.std(current_returns))
            best_label = "modelo_actual"
            best_path = final_zip_path
            best_model_data = (current_returns, current_mean, current_std, "modelo_actual")

            print(f"\nModelo actual:")
            print(f"Media: {current_mean:.2f} ± {current_std:.2f}")
            print(f"Retornos: {current_returns}")

            for label, path in candidate_paths:
                candidate_model = _load_dqn(path) # Cargar con la clase DQN
                candidate_returns = []
                for _ in range(comparison_episodes):
                    obs = eval_env.reset()
                    done = False
                    episode_return = 0.0
                    while not done:
                        action, _ = candidate_model.predict(obs, deterministic=True)
                        obs, rewards, dones, _ = eval_env.step(action)
                        episode_return += float(rewards[0])
                        done = bool(dones[0])
                    candidate_returns.append(episode_return)

                candidate_mean = float(np.mean(candidate_returns))
                candidate_std = float(np.std(candidate_returns))
                
                print(f"\n {label}:")
                print(f"Media: {candidate_mean:.2f} ± {candidate_std:.2f}")
                print(f"Retornos: {candidate_returns}")
                
                t_stat, p_value = stats.ttest_ind(candidate_returns, current_returns, alternative='greater')
                cohens_d = (candidate_mean - current_mean) / np.sqrt((current_std**2 + candidate_std**2) / 2) if (current_std**2 + candidate_std**2) > 0 else 0
                
                print(f"     t-test (candidato > actual): t={t_stat:.4f}, p={p_value:.4f}, Cohen's d={cohens_d:.4f}")
                
                if p_value < 0.3 and candidate_mean > current_mean:
                    print(f"Significativamente mejor (p < 0.3)")
                    best_label = label
                    best_path = path
                    best_model_data = (candidate_returns, candidate_mean, candidate_std, label)
                elif candidate_mean > current_mean:
                    print(f"AVISO: Mejor media pero NO significativo (p ≥ 0.3)")
                else:
                    print(f"No es mejor que el actual")

            print(f"\n[resultado] Mejor modelo elegido: {best_label}")
            if os.path.abspath(best_path) != os.path.abspath(final_zip_path):
                shutil.copy2(best_path, final_zip_path)
                returns, mean, std, name = best_model_data
                print(f"Sobreescrito best_model. Media: {mean:.2f} ± {std:.2f}")
            else:
                returns, mean, std, name = best_model_data
                print(f"Modelo actual es el mejor. Mantiene su posición. Media: {mean:.2f} ± {std:.2f}")
        except Exception as e:
            print(f"AVISO: Error en comparación estadística: {e}. Se mantiene el modelo actual.")
            import traceback
            traceback.print_exc()
    else:
        model.save(model_path)
        print(f"No hay candidatos evaluables para comparar; se guardó el último modelo en: {final_zip_path}")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()