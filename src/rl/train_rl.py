import argparse
import os
from gym_env import SpaceInvadersGymEnv


from stable_baselines3.common.monitor import Monitor

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


if __name__ == "__main__":
    main()