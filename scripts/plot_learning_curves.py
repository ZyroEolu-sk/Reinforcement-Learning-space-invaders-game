from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _collect_evaluation_files(paths: list[str]) -> list[Path]:
    if not paths:
        files = sorted(PROJECT_ROOT.rglob("evaluations.npz"))
        return files

    files: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = _resolve_path(raw_path)
        if path.is_dir():
            candidates = sorted(path.rglob("evaluations.npz"))
        else:
            candidates = [path]

        for candidate in candidates:
            if candidate.is_file() and candidate not in seen:
                files.append(candidate)
                seen.add(candidate)

    return files


def _series_label(path: Path) -> str:
    try:
        return str(path.parent.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.parent)


def _load_evaluation_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    required_keys = {"timesteps", "results", "ep_lengths"}
    missing_keys = required_keys.difference(data.files)
    if missing_keys:
        raise ValueError(f"{path} is missing keys: {sorted(missing_keys)}")

    timesteps = np.asarray(data["timesteps"], dtype=float)
    results = np.asarray(data["results"], dtype=float)
    ep_lengths = np.asarray(data["ep_lengths"], dtype=float)
    return timesteps, results, ep_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning curves from Stable-Baselines3 evaluations.npz files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="evaluation files or directories to scan recursively (defaults to all evaluations.npz in the repo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/learning_curves.png",
        help="Output image path (relative to the project root).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Learning Curves",
        help="Figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = _collect_evaluation_files(args.paths)

    if not files:
        raise FileNotFoundError("No evaluations.npz files found.")

    fig, (ax_reward, ax_length) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(args.title)

    for path in files:
        timesteps, results, ep_lengths = _load_evaluation_data(path)
        label = _series_label(path)

        reward_mean = results.mean(axis=1)
        reward_std = results.std(axis=1)
        length_mean = ep_lengths.mean(axis=1)
        length_std = ep_lengths.std(axis=1)

        ax_reward.plot(timesteps, reward_mean, label=label)
        ax_reward.fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.15)

        ax_length.plot(timesteps, length_mean, label=label)
        ax_length.fill_between(timesteps, length_mean - length_std, length_mean + length_std, alpha=0.15)

    ax_reward.set_ylabel("Mean reward")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(loc="best")

    ax_length.set_xlabel("Timesteps")
    ax_length.set_ylabel("Mean episode length")
    ax_length.grid(True, alpha=0.3)
    ax_length.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()