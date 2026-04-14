import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from gym_env_vision import SpaceInvadersVisionEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def make_play_env_fn(args):
    def _factory():
        return Monitor(
            SpaceInvadersVisionEnv(
                render_mode="human",
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
    parser = argparse.ArgumentParser(description="Run a trained vision PPO agent for Space Invaders.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/vision/best_model/best_model",
        help="Path to the trained model (with or without .zip).",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Frame skip used by the environment.")
    parser.add_argument("--max-steps", type=int, default=12000, help="Maximum steps per episode.")
    parser.add_argument("--img-width", type=int, default=None, help="Observation frame width after preprocessing (auto if omitted).")
    parser.add_argument("--img-height", type=int, default=None, help="Observation frame height after preprocessing (auto if omitted).")
    parser.add_argument("--start-level", type=int, default=1, help="Starting level (1-4).")
    parser.add_argument("--max-level", type=int, default=4, help="Maximum level target (1-4).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default is deterministic).",
    )
    return parser.parse_args()


def _infer_image_size_from_model(model: PPO) -> tuple[int, int]:
    shape = getattr(model.observation_space, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError(f"Unsupported observation shape in model: {shape}")

    # Expected vision model shapes are CHW=(4, H, W) or HWC=(H, W, 4).
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


def main():
    args = parse_args()

    model_path = _resolve_project_path(args.model_path)
    if model_path.suffix != ".zip":
        model_file = model_path.with_suffix(".zip")
    else:
        model_file = model_path

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model_probe = PPO.load(str(model_file))
    inferred_width, inferred_height = _infer_image_size_from_model(model_probe)
    args.enable_combo_actions = _infer_combo_actions_from_model(model_probe)
    args.img_width = inferred_width if args.img_width is None else int(args.img_width)
    args.img_height = inferred_height if args.img_height is None else int(args.img_height)

    env = VecTransposeImage(DummyVecEnv([make_play_env_fn(args)]))
    model = PPO.load(str(model_file), env=env)

    print(f"Loaded model: {model_file}")
    print(f"Using observation size: {args.img_width}x{args.img_height}")
    print(f"Using combo actions: {args.enable_combo_actions}")
    print(f"Playing {args.episodes} episode(s)...")

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, dones, infos = env.step(action)

            episode_reward += float(reward[0])
            episode_steps += 1
            done = bool(dones[0])
            info = infos[0]

        print(
            f"Episode {episode}: reward={episode_reward:.2f}, "
            f"steps={episode_steps}, score={info.get('score', 'n/a')}, "
            f"level={info.get('level', 'n/a')}, completed_game={info.get('completed_game', False)}"
        )

    env.close()


if __name__ == "__main__":
    main()
