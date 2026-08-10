import argparse
import os
import shutil
from pathlib import Path
from typing import Callable
import numpy as np
import math
from scipy import stats

from gym_env_vision import SpaceInvadersVisionEnv
from custom_cnn import SpaceInvadersResidualSiluCNN

from stable_baselines3 import PPO
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
        description="Continue training from an existing vision PPO model for Space Invaders."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/vision/best_model/best_model",
        help="Path to an existing PPO model (.zip optional).",
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
        help="Directory where continued-training outputs will be saved (relative to project root).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel training environments.")
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
    parser.add_argument(
        "--override-learning-rate",
        type=float,
        default=3e-4,
        help="Override learning rate for continued training (e.g. 3e-4).",
    )
    parser.add_argument(
        "--override-ent-coef",
        type=float,
        default=0.03,
        help="Override entropy coefficient to increase exploration (e.g. 0.02-0.05).",
    )
    parser.add_argument(
        "--override-clip-range",
        type=float,
        default=0.25,
        help="Override PPO clip range (e.g. 0.15-0.25).",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="linear",
        choices=["constant", "linear", "cosine", "polynomial"],
        help="Learning-rate schedule to use when continuing training (default: linear).",
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

    raise FileNotFoundError(
        f"No se encontro el modelo en '{path}' ni en '{zip_path}'."
    )


def _load_ppo(path: str, **kwargs) -> PPO:
    """Load PPO models with support for custom feature extractors."""
    return PPO.load(path, custom_objects=CUSTOM_OBJECTS, **kwargs)


def _assert_residual_extractor(model: PPO, model_path: str) -> None:
    extractor_name = type(model.policy.features_extractor).__name__
    if extractor_name != "SpaceInvadersResidualSiluCNN":
        raise ValueError(
            "Este script esta forzado al extractor personalizado SpaceInvadersResidualSiluCNN. "
            f"El modelo cargado desde '{model_path}' usa '{extractor_name}'. "
            "Entrena/continua con un checkpoint residual en models/vision."
        )
    print(f"[model] extractor={extractor_name}")

def _infer_image_size_from_model(model: PPO) -> tuple[int, int]:
    shape = getattr(model.observation_space, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError(f"Unsupported observation shape in model: {shape}")

    if shape[0] == 4:
        return int(shape[2]), int(shape[1])
    if shape[2] == 4:
        return int(shape[1]), int(shape[0])

    raise ValueError(f"Cannot infer image size from observation shape: {shape}")


def _infer_combo_actions_from_model(model: PPO) -> bool:
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


def _apply_hyperparameter_overrides(model: PPO, args) -> None:
    # Anti-plateau defaults: use stronger exploration and faster updates unless explicitly overridden.
    lr = float(args.override_learning_rate) if args.override_learning_rate is not None else 2e-4
    # Apply schedule selection
    if getattr(args, "lr_schedule", "linear") == "constant":
        model.learning_rate = lr
        model.lr_schedule = get_schedule_fn(lr)
    elif args.lr_schedule == "linear":
        model.learning_rate = lr
        model.lr_schedule = linear_schedule(lr)
    elif args.lr_schedule == "cosine":
        model.learning_rate = lr
        model.lr_schedule = cosine_schedule(lr)
    else:
        model.learning_rate = lr
        model.lr_schedule = get_schedule_fn(lr)

    if args.override_learning_rate is not None:
        print(f"[override] learning_rate={lr} schedule={args.lr_schedule}")
    else:
        print(f"[default anti-plateau] learning_rate={lr} schedule={args.lr_schedule}")

    ent_coef = float(args.override_ent_coef) if args.override_ent_coef is not None else 0.05
    model.ent_coef = ent_coef
    if args.override_ent_coef is not None:
        print(f"[override] ent_coef={ent_coef}")
    else:
        print(f"[default anti-plateau] ent_coef={ent_coef}")

    if args.override_clip_range is not None:
        clip_range = float(args.override_clip_range)
        model.clip_range = get_schedule_fn(clip_range)
        print(f"[override] clip_range={clip_range}")


def _select_batch_size(rollout_size: int, preferred: int = 512) -> int:
    """Pick the largest batch <= preferred that exactly divides rollout_size."""
    if rollout_size <= 0:
        raise ValueError(f"rollout_size must be > 0, got {rollout_size}")
    upper = min(preferred, rollout_size)
    for size in range(upper, 0, -1):
        if rollout_size % size == 0:
            return size
    return 1


def main():
    args = parse_args()
    print("[forced-custom-cnn] Continue mode requires SpaceInvadersResidualSiluCNN checkpoints.")

    model_path = _resolve_model_path(args.model_path)
    model_probe = _load_ppo(model_path)
    _assert_residual_extractor(model_probe, model_path)
    inferred_width, inferred_height = _infer_image_size_from_model(model_probe)
    args.enable_combo_actions = _infer_combo_actions_from_model(model_probe)
    args.img_width = inferred_width if args.img_width is None else int(args.img_width)
    args.img_height = inferred_height if args.img_height is None else int(args.img_height)

    save_dir = _resolve_project_path(args.save_dir)

    os.makedirs(save_dir, exist_ok=True)
    # Keep checkpoints/logs separated and store eval candidates in a dedicated folder.
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
        name_prefix="ppo_space_invaders_vision_continued",
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

    model = _load_ppo(
        model_path,
        env=train_env,
        tensorboard_log=logs_dir,
        seed=args.seed,
    )
    _assert_residual_extractor(model, model_path)
    _apply_hyperparameter_overrides(model, args)
    rollout_size = int(model.n_steps) * num_envs
    model.batch_size = _select_batch_size(rollout_size, preferred=512)
    print(f"[config] n_steps={model.n_steps} num_envs={num_envs} rollout={rollout_size} batch={model.batch_size}")

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

    # Compare current best model against available candidates (eval best and/or interrupted snapshot)
    # and only overwrite if a candidate is better.
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
            current_best = _load_ppo(final_zip_path)

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
                candidate_model = _load_ppo(path)
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
                
                # T-test: ¿Es el candidato estadísticamente mejor?
                t_stat, p_value = stats.ttest_ind(candidate_returns, current_returns, alternative='greater')
                cohens_d = (candidate_mean - current_mean) / np.sqrt((current_std**2 + candidate_std**2) / 2) if (current_std**2 + candidate_std**2) > 0 else 0
                
                print(f"     t-test (candidato > actual): t={t_stat:.4f}, p={p_value:.4f}, Cohen's d={cohens_d:.4f}")
                
                # Decidir si reemplazar basado en p-value < 0.15
                if p_value < 0.15 and candidate_mean > current_mean:
                    print(f"Significativamente mejor (p < 0.15)")
                    best_label = label
                    best_path = path
                    best_model_data = (candidate_returns, candidate_mean, candidate_std, label)
                elif candidate_mean > current_mean:
                    print(f"AVISO: Mejor media pero NO significativo (p ≥ 0.15)")
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
        # No candidates available or final model does not exist: save the latest in-memory model.
        model.save(model_path)
        print(
            "No hay candidatos evaluables para comparar; "
            f"se guardo el último modelo entrenado en: {final_zip_path}"
        )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()