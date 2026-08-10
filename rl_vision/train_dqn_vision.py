import argparse
import os
import shutil
from pathlib import Path
from typing import Callable
import math
import numpy as np
import cv2
import torch
from torch import nn
from gym_env_dqn import SpaceInvadersDQNEnv as SpaceInvadersVisionEnv
from custom_cnn import SpaceInvadersResidualSiluCNN, SpaceInvadersSimpleSiluCNN

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3 import DQN

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


class ObservationDebugCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._saved = 0

    def _save_observation(self, obs: np.ndarray, overlay_lines: list[str]):
        if obs.ndim == 4:
            frame_stack = obs[0]
        else:
            frame_stack = obs

        if frame_stack.ndim == 3 and frame_stack.shape[0] == 4:
            channels = frame_stack
        elif frame_stack.ndim == 3 and frame_stack.shape[-1] == 4:
            channels = np.transpose(frame_stack, (2, 0, 1))
        else:
            return

        h, w = channels.shape[1], channels.shape[2]
        tiled = np.zeros((h * 2, w * 2), dtype=np.uint8)
        tiled[0:h, 0:w] = channels[0]
        tiled[0:h, w:2 * w] = channels[1]
        tiled[h:2 * h, 0:w] = channels[2]
        tiled[h:2 * h, w:2 * w] = channels[3]

        out_path = os.path.join(self.save_dir, f"obs_{self.num_timesteps:08d}.png")
        cv2.imwrite(out_path, tiled)

        overlay = cv2.cvtColor(tiled, cv2.COLOR_GRAY2BGR)
        y = 16
        for line in overlay_lines:
            cv2.putText(
                overlay,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 16

        overlay_path = os.path.join(self.save_dir, f"obs_{self.num_timesteps:08d}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        self._saved += 1

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        obs = self.locals.get("new_obs", None)
        if obs is None:
            return True

        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        env0_info = infos[0] if len(infos) > 0 else {}
        env0_reward = float(rewards[0]) if len(rewards) > 0 else 0.0

        obs_arr = obs[0] if getattr(obs, "ndim", 0) == 4 else obs
        obs_min = int(np.min(obs_arr))
        obs_max = int(np.max(obs_arr))
        obs_mean = float(np.mean(obs_arr))

        overlay_lines = [
            f"t={self.num_timesteps} saved={self._saved + 1}",
            f"reward={env0_reward:.3f} score={env0_info.get('score', 'n/a')} level={env0_info.get('level', 'n/a')}",
            f"lives={env0_info.get('lives', 'n/a')} done={env0_info.get('game_over', 'n/a')}",
            f"pix min/max/mean={obs_min}/{obs_max}/{obs_mean:.1f}",
        ]

        self._save_observation(obs, overlay_lines)
        if self.verbose > 0:
            print(f"[obs-debug] saved={self._saved} at timesteps={self.num_timesteps}")
        return True


def make_env_fn(args, is_eval: bool = False) -> Callable[[], Monitor]:
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
            )
        )
    return _factory


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent to play Space Invaders using DQN.")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total number of timesteps for training.")
    parser.add_argument("--save-dir", type=str, default="models/vision", help="Directory to save the trained model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Frames to skip per action.")
    parser.add_argument("--max-steps", type=int, default=12000, help="Maximum steps per episode.")
    parser.add_argument("--img-width", type=int, default=96, help="Observation frame width after preprocessing.")
    parser.add_argument("--img-height", type=int, default=112, help="Observation frame height after preprocessing.")
    parser.add_argument("--start-level", type=int, default=1, help="Curriculum initial level.")
    parser.add_argument("--max-level", type=int, default=1, help="Curriculum target level.")
    parser.add_argument("--eval-freq", type=int, default=25000, help="Evaluation frequency in timesteps.")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Episodes per periodic evaluation.")
    parser.add_argument("--checkpoint-freq", type=int, default=25000, help="Checkpoint frequency in timesteps.")
    parser.add_argument("--obs-debug-freq", type=int, default=0, help="Save stacked observation images.")
    parser.add_argument("--obs-debug-dir", type=str, default="models/obs_debug", help="Directory for observation debug images.")
    
    # Parámetros específicos de DQN y arquitectura
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DQN training.")
    parser.add_argument(
        "--preset",
        type=str,
        default="baseline",
        choices=["baseline", "conservative", "explore", "optimized"],
        help="DQN Hyperparameter presets.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "polynomial"],
        help="Learning-rate schedule.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="residual",
        choices=["residual", "simple"],
        help="Feature extractor architecture.",
    )
    return parser.parse_args()


def _get_hparams_from_preset(preset: str) -> dict:
    if preset == "conservative":
        return {
            "learning_rate": 5e-5,
            "buffer_size": 50000,
            "learning_starts": 5000,
            "target_update_interval": 2000,
        }
    if preset == "explore":
        return {
            "learning_rate": 2e-4,
            "buffer_size": 100000,
            "learning_starts": 10000,
            "target_update_interval": 1000,
        }
    if preset == "optimized":
        return {
            "learning_rate": 3e-4,
            "buffer_size": 80000,
            "learning_starts": 5000,
            "target_update_interval": 1000,
        }
    return {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 10000,
        "target_update_interval": 1000,
    }


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    return lambda progress_remaining: progress_remaining * initial_value


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    return lambda progress_remaining: 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining))) * initial_value


def polynomial_schedule(initial_value: float, power: float = 1.0) -> Callable[[float], float]:
    return lambda progress_remaining: (progress_remaining ** power) * initial_value

def _resolve_project_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _build_policy_kwargs(arch: str) -> dict:
    if arch == "simple":
        return {
            "features_extractor_class": SpaceInvadersSimpleSiluCNN,
            "features_extractor_kwargs": {"features_dim": 320},
            "net_arch": [512, 256],
            "activation_fn": nn.SiLU,
        }
    return {
        "features_extractor_class": SpaceInvadersResidualSiluCNN,
        "features_extractor_kwargs": {"features_dim": 512, "dropout_p": 0.08},
        "net_arch": [512, 256],
        "activation_fn": nn.SiLU,
    }

def main():
    args = parse_args()
    preset_hparams = _get_hparams_from_preset(args.preset)
    
    # Identificador descriptivo y único para este experimento
    lr_value = preset_hparams["learning_rate"]
    model_identifier = f"dqn_{args.arch}_{args.preset}_lr{lr_value:.0e}_b{args.batch_size}"
    # Reemplazamos caracteres conflictivos (ej. 1e-04 -> 1e-04, quitamos signos raros)
    model_identifier = model_identifier.replace("-", "")

    # Modificamos el subdirectorio para que este experimento tenga sus propios logs y checkpoints separados
    base_save_dir = _resolve_project_path(args.save_dir)
    save_dir = os.path.join(base_save_dir, model_identifier)

    os.makedirs(save_dir, exist_ok=True)
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    best_model_dir = os.path.join(save_dir, "best_model")
    logs_dir = os.path.join(save_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    num_envs = max(1, args.num_envs)
    train_factories = [make_env_fn(args) for _ in range(num_envs)]
    train_env_base = DummyVecEnv(train_factories)
    eval_env_base = DummyVecEnv([make_env_fn(args, is_eval=True)])
    train_env = VecTransposeImage(train_env_base)
    eval_env = VecTransposeImage(eval_env_base)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoints_dir,
        name_prefix=model_identifier, # Checkpoints usarán el nombre descriptivo
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=logs_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=max(1, int(args.n_eval_episodes)),
        deterministic=True,
        render=False,
    )
    clear_rate_callback = ClearRateCallback()
    callbacks = [checkpoint_callback, eval_callback, clear_rate_callback]

    if args.obs_debug_freq > 0:
        obs_debug_dir = os.path.join(save_dir, "obs_debug")
        callbacks.append(ObservationDebugCallback(args.obs_debug_freq, obs_debug_dir, verbose=1))

    policy_kwargs = _build_policy_kwargs(args.arch)

    # Esquema de Learning Rate
    initial_lr = preset_hparams["learning_rate"]
    if args.lr_schedule == "constant":
        lr = initial_lr
    elif args.lr_schedule == "linear":
        lr = linear_schedule(initial_lr)
    elif args.lr_schedule == "cosine":
        lr = cosine_schedule(initial_lr)
    elif args.lr_schedule == "polynomial":
        lr = polynomial_schedule(initial_lr, power=1.5)
    else:
        lr = initial_lr

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 70)
    print(f"INICIANDO EXPERIMENTO: {model_identifier.upper()}")
    print(f"Guardando todo en: {save_dir}")
    print(f"Hardware: {device.upper()} | Envs: {num_envs} | LR Schedule: {args.lr_schedule}")
    print("=" * 70)

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=lr,
        buffer_size=preset_hparams["buffer_size"],
        learning_starts=preset_hparams["learning_starts"],
        batch_size=args.batch_size, # Pasamos el argumento dinámico de la consola
        tau=1.0,
        target_update_interval=preset_hparams["target_update_interval"],
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.15,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        max_grad_norm=10,
        verbose=1,
        tensorboard_log=logs_dir,
        seed=args.seed,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # El archivo .zip final ahora llevará incorporado el nombre del experimento descriptivo
    final_zip_name = f"{model_identifier}.zip"
    final_zip_path = os.path.join(best_model_dir, final_zip_name)
    eval_best_model_path = os.path.join(best_model_dir, "best_model.zip")

    if os.path.isfile(eval_best_model_path):
        shutil.move(eval_best_model_path, final_zip_path)
        print(f"¡Éxito! El mejor modelo se guardó como: {final_zip_path}")
    else:
        model.save(os.path.join(best_model_dir, model_identifier))
        print(f"AVISO: Guardado el último modelo de la memoria como: {final_zip_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()