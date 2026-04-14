import argparse
import os
import shutil
from pathlib import Path
from typing import Callable

from gym_env import SpaceInvadersGymEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


class ClearRateCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.completed_episodes = 0
        self.cleared_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                self.completed_episodes += 1
                if info.get("completed_game", False):
                    self.cleared_episodes += 1

        if self.completed_episodes > 0:
            clear_rate = self.cleared_episodes / self.completed_episodes
            self.logger.record("rollout/clear_rate", clear_rate)

        return True


def make_env_fn(args) -> Callable[[], Monitor]:
    def _factory():
        return Monitor(
            SpaceInvadersGymEnv(
                render_mode=None,
                max_steps=args.max_steps,
                frame_skip=args.frame_skip,
                start_level=args.start_level,
                max_level=args.max_level,
            )
        )

    return _factory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training from an existing PPO model for Space Invaders."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/vector/best_model/best_model",
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
        default="models/vector",
        help="Directory where continued-training outputs will be saved (relative to project root).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-envs", type=int, default=12, help="Parallel training environments.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Frames to skip per action.")
    parser.add_argument("--max-steps", type=int, default=12000, help="Max steps per episode.")
    parser.add_argument("--start-level", type=int, default=1, help="Curriculum initial level (1-4).")
    parser.add_argument("--max-level", type=int, default=4, help="Curriculum target level (1-4).")
    parser.add_argument("--eval-freq", type=int, default=25000, help="Evaluation frequency in timesteps.")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency in timesteps.")
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


def main():
    args = parse_args()

    model_path = _resolve_model_path(args.model_path)
    save_dir = _resolve_project_path(args.save_dir)

    os.makedirs(save_dir, exist_ok=True)
    # Keep checkpoints/logs separated, but keep a single best_model directory.
    checkpoints_dir = os.path.join(save_dir, "checkpoints_continued")
    logs_dir = os.path.join(save_dir, "logs_continued")
    best_model_dir = os.path.dirname(model_path)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    num_envs = max(1, args.num_envs)
    train_factories = [make_env_fn(args) for _ in range(num_envs)]
    train_env = SubprocVecEnv(train_factories) if num_envs > 1 else DummyVecEnv(train_factories)
    eval_env = DummyVecEnv([make_env_fn(args)])

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="ppo_space_invaders_continued",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    clear_rate_callback = ClearRateCallback()

    model = PPO.load(
        model_path,
        env=train_env,
        tensorboard_log=logs_dir,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.additional_timesteps,
        callback=[checkpoint_callback, eval_callback, clear_rate_callback],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Compare new best model from eval callbacks with current best model
    # Only overwrite if the new model is better
    final_zip_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
    eval_best_model_path = os.path.join(best_model_dir, "best_model.zip")
    
    if os.path.isfile(eval_best_model_path) and os.path.abspath(eval_best_model_path) != os.path.abspath(final_zip_path):
        print("\n[model-comparison] Comparando modelo actual vs nuevo modelo entrenado...")
        try:
            # Load both models for comparison
            current_best = PPO.load(final_zip_path)
            new_candidate = PPO.load(eval_best_model_path)
            
            # Run quick evaluations (5 episodes each) to compare
            print("  Evaluando modelo actual (5 episodios)...")
            current_returns = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                done = False
                episode_return = 0.0
                while not done:
                    action, _ = current_best.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = eval_env.step(action)
                    episode_return += float(reward)
                    done = done or truncated
                current_returns.append(episode_return)
            
            print("  Evaluando modelo nuevo (5 episodios)...")
            new_returns = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                done = False
                episode_return = 0.0
                while not done:
                    action, _ = new_candidate.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = eval_env.step(action)
                    episode_return += float(reward)
                    done = done or truncated
                new_returns.append(episode_return)
            
            current_mean = float(np.mean(current_returns))
            new_mean = float(np.mean(new_returns))
            
            print(f"  Modelo actual: {current_mean:.2f} (returnos: {current_returns})")
            print(f"  Modelo nuevo:  {new_mean:.2f} (returnos: {new_returns})")
            
            if new_mean > current_mean:
                shutil.copy2(eval_best_model_path, final_zip_path)
                print(f"✓ Modelo nuevo es mejor (+{new_mean - current_mean:.2f}). Sobreescrito: {final_zip_path}")
            else:
                print(f"✗ Modelo actual es mejor o igual. Manteniéndolo: {final_zip_path}")
        except Exception as e:
            print(f"  ⚠ Error en comparación: {e}. Usando EvalCallback best_model.")
            shutil.copy2(eval_best_model_path, final_zip_path)
    elif os.path.isfile(eval_best_model_path):
        # Same file path, no copy needed
        print(f"✓ Mejor modelo guardado en: {final_zip_path}")
    else:
        # No eval best model found, save current
        model.save(model_path)
        print(
            "⚠ No se encontro best_model.zip en evaluaciones; "
            f"se guardo el último modelo entrenado en: {final_zip_path}"
        )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
