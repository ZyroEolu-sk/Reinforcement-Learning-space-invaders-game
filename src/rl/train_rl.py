import argparse
import os
from typing import Callable
from gym_env import SpaceInvadersGymEnv


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3 import PPO


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


def make_env_fn(args, is_eval: bool = False) -> Callable[[], Monitor]:
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
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent to play Space Invaders.")
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Total number of timesteps for training.")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save the trained model.")
    parser.add_argument("--model-name", type=str, default="space_invaders_agent", help="Name of the saved model file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-envs", type=int, default=12, help="Number of parallel environments for training.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Number of skipped frames in the environment.")
    parser.add_argument("--max-steps", type=int, default=12000, help="Maximum number of steps per episode.")
    parser.add_argument("--start-level", type=int, default=1, help="Curriculum initial level (1-4).")
    parser.add_argument("--max-level", type=int, default=4, help="Curriculum target level (1-4).")
    parser.add_argument("--eval-freq", type=int, default=25000, help="Evaluation frequency in timesteps.")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency in timesteps.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create necessary directories for saving models and logs
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.save_dir, "checkpoints")
    best_model_dir = os.path.join(args.save_dir, "best_model")
    logs_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    num_envs = max(1, args.num_envs)
    train_factories = [make_env_fn(args) for _ in range(num_envs)]
    train_env = SubprocVecEnv(train_factories) if num_envs > 1 else DummyVecEnv(train_factories)
    eval_env = DummyVecEnv([make_env_fn(args, is_eval=True)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix="ppo_space_invaders",
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

    n_steps = 1024
    rollout_size = n_steps * num_envs
    batch_size = 512 if rollout_size >= 512 else rollout_size

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_dir,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback, clear_rate_callback],
        progress_bar=True,
    )

    final_path = os.path.join(args.save_dir, args.model_name)
    model.save(final_path)
    print(f"Modelo final guardado en: {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()