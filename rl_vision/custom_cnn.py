import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _group_count(channels: int) -> int:
    """Pick a GroupNorm group count that divides the channel count."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SqueezeExcitation(nn.Module):
    """Channel attention block (SE) for improved feature recalibration."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class DropPath(nn.Module):
    """Per-sample stochastic depth regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class ResidualConvBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, Dropout2d and SE attention."""

    def __init__(self, channels: int, dropout_p: float = 0.10, drop_path_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout_p)
        self.se = SqueezeExcitation(channels)
        self.drop_path = DropPath(drop_path_p)

    def forward(self, x: th.Tensor) -> th.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.drop_path(out)
        out = out + identity
        out = self.act(out)
        return out


class DownsampleStage(nn.Module):
    """Downsampling stage followed by residual refinement blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        dropout_p: float,
        drop_path_start: float,
        drop_path_end: float,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        ]
        if num_blocks == 1:
            drop_rates = [drop_path_end]
        else:
            drop_rates = [
                drop_path_start + (drop_path_end - drop_path_start) * i / (num_blocks - 1)
                for i in range(num_blocks)
            ]
        for drop_rate in drop_rates:
            layers.append(ResidualConvBlock(out_channels, dropout_p=dropout_p, drop_path_p=drop_rate))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.stage(x)


class SpaceInvadersResidualSiluCNN(BaseFeaturesExtractor):
    """Deep residual CNN with SE + GroupNorm + SiLU for PPO vision policies."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 384,
        dropout_p: float = 0.10,
        drop_path_rate: float = 0.10,
    ):
        super().__init__(observation_space, features_dim)
        assert hasattr(observation_space, "shape") and len(observation_space.shape) == 3
        in_channels = observation_space.shape[0]

        stage_drop = {
            "s1": (0.00, drop_path_rate * 0.30),
            "s2": (drop_path_rate * 0.20, drop_path_rate * 0.50),
            "s3": (drop_path_rate * 0.40, drop_path_rate * 0.70),
            "s4": (drop_path_rate * 0.60, drop_path_rate * 0.85),
            "s5": (drop_path_rate * 0.80, drop_path_rate),
        }

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(48), 48),
            nn.SiLU(),
            ResidualConvBlock(48, dropout_p=dropout_p, drop_path_p=0.0),
            DownsampleStage(48, 72, num_blocks=2, dropout_p=dropout_p, drop_path_start=stage_drop["s1"][0], drop_path_end=stage_drop["s1"][1]),
            DownsampleStage(72, 96, num_blocks=2, dropout_p=dropout_p, drop_path_start=stage_drop["s2"][0], drop_path_end=stage_drop["s2"][1]),
            DownsampleStage(96, 128, num_blocks=2, dropout_p=dropout_p, drop_path_start=stage_drop["s3"][0], drop_path_end=stage_drop["s3"][1]),
            DownsampleStage(128, 160, num_blocks=2, dropout_p=dropout_p, drop_path_start=stage_drop["s4"][0], drop_path_end=stage_drop["s4"][1]),
            DownsampleStage(160, 224, num_blocks=1, dropout_p=dropout_p, drop_path_start=stage_drop["s5"][0], drop_path_end=stage_drop["s5"][1]),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(224, features_dim),
            nn.SiLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.backbone(observations)
        return self.head(x)


class SpaceInvadersSimpleSiluCNN(BaseFeaturesExtractor):
    """Lightweight baseline CNN extractor for architecture ablations."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        assert hasattr(observation_space, "shape") and len(observation_space.shape) == 3
        in_channels = observation_space.shape[0]

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.SiLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.backbone(observations)
        return self.head(x)
