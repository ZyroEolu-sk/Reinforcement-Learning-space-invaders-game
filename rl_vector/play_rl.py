import argparse
from pathlib import Path

from stable_baselines3 import PPO

from gym_env import SpaceInvadersGymEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())

def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained reinforcement learning agent in Space Invaders.")
    parser.add_argument("--model-path", type=str, default="models/vector/best_model/best_model", help="Path to the trained model file (relative to project root).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to play.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = _resolve_project_path(args.model_path)

    env = SpaceInvadersGymEnv(render_mode="human", max_steps=12000, frame_skip=1, start_level=1, max_level=4)
    model = PPO.load(model_path)
    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"Episodio {episode + 1}: reward_total={total_reward:.2f} "
                f"score={info.get('score', 0)} lives={info.get('lives', 0)}"
            )
    env.close()

if __name__ == "__main__":
    main()