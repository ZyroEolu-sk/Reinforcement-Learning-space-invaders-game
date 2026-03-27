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
        default="models/best_model/best_model",
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
        default="models",
        help="Directory where continued-training outputs will be saved (relative to project root).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="best_model",
        help="Name for the final continued model file.",
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
    checkpoints_dir = os.path.join(save_dir, "checkpoints_continued")
    best_model_dir = os.path.join(save_dir, "best_model_continued")
    logs_dir = os.path.join(save_dir, "logs_continued")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

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

    final_path = os.path.join(save_dir, args.model_name)
    final_zip_path = final_path if final_path.endswith(".zip") else f"{final_path}.zip"
    best_model_path = os.path.join(best_model_dir, "best_model.zip")

    if os.path.isfile(best_model_path):
        shutil.copy2(best_model_path, final_zip_path)
        print(f"Modelo con mejor recompensa guardado en: {final_zip_path}")
    else:
        model.save(final_path)
        print(
            "No se encontro best_model.zip (posible falta de evaluaciones); "
            f"se guardo el ultimo modelo en: {final_zip_path}"
        )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
