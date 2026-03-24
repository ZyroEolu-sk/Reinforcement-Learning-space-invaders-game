from typing import Optional

import gymnasium as gym
from src.main import Game

import numpy as np

class SpaceInvadersGymEnv(gym.Env):
    """Gym environment for Space Invaders.
    
    Actions:
    0: no-op
    1: move left
    2: move right
    3: shoot
    """
    # Metadata for rendering
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 4500, frame_skip: int = 2):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip

        self.action_space = gym.spaces.Discrete(4)  # 4 discrete actions
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(20, ), 
            dtype=np.float32
        )  

        self.game = Game()
        self.steps = 0
        self._last_score = 0
        self._last_lives = 0