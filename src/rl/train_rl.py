import argparse
import os
from gym_env import SpaceInvadersGymEnv


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO


def make_train_env():
    return Monitor(SpaceInvadersGymEnv(render_mode=None, max_steps=4500, frame_skip=2))

def make_eval_env():
    return Monitor(SpaceInvadersGymEnv(render_mode=None, max_steps=4500, frame_skip=2))

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent to play Space Invaders.")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total number of timesteps for training.")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save the trained model.")
    parser.add_argument("--model-name", type=str, default="space_invaders_agent", help="Name of the saved model file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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

    # Create training and evaluation environments (using DummyVecEnv for simplicity, less overhead than SubprocVecEnv)
    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    #Create training and evaluation environments (using SubprocVecEnv for better performance with multiple environments)
    # num_envs = 4  # Number of parallel environments for training
    # train_env = SubprocVecEnv([make_train_env for _ in range(num_envs)])
    # eval_env = SubprocVecEnv([make_eval_env for _ in range(num_envs)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=checkpoints_dir,
        name_prefix="ppo_space_invaders",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_dir,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(args.save_dir, args.model_name)
    model.save(final_path)
    print(f"Modelo final guardado en: {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()