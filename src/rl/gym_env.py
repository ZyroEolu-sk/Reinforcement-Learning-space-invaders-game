import random
from typing import Optional


import sys
import os

# Obtain the current directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/rl
parent_dir = os.path.dirname(current_dir)                # .../src

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from main import Game
from effects import Bullet
from settings import PLAYER_VEL, WINDOW_WIDTH, RED, BULLET_VEL

import gymnasium as gym
from main import Game

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
        """
        Initializes the Space Invaders Gym environment.
        Args:
            render_mode (str, optional): The mode to render the environment. Defaults to None.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 4500.
            frame_skip (int, optional): Number of frames to skip between actions. Defaults to 2.
        """
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

    
    def _apply_action(self, action: int):
        """Applies the given action to the game state."""
        if self.game.cooldown > 0:
            self.game.cooldown -= 1

        if action == 1 and self.game.player.x > 0:  # Move left
            self.game.player.x -= PLAYER_VEL
        
        elif action == 2 and self.game.player.x < WINDOW_WIDTH - 60:
            self.game.player.x += PLAYER_VEL
        elif action == 3 and self.game.cooldown == 0 and self.game.player_is_alive:  # Shoot
            self.game.bullets_group.add(
                Bullet(
                    -1,
                    (self.game.player.centerx, self.game.player.y - 15),
                    RED,
                    5, 
                    17, 
                    BULLET_VEL, 
                    0,
                )
            )
            self.game.cooldown = 40

