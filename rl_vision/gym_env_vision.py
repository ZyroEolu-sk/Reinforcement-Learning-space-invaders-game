from typing import Optional

import gymnasium as gym
import numpy as np
from collections import deque
import cv2

import sys
import os
import random
import pygame

# Obtain the current directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../rl
game_src_dir = os.path.join(os.path.dirname(current_dir), "space-invaders-game", "src")
# Add the game source directory to sys.path if it's not already there
if game_src_dir not in sys.path:
    sys.path.append(game_src_dir)

from main import Game
from settings import WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_VEL, RED, BLACK, BULLET_VEL
from entities import TechAlien, Braincell
from effects import Bullet

class SpaceInvadersVisionEnv(gym.Env):
    """
    Gym environment for Space Invaders using vision-based observations.
    
    Actions:
    0: no-op
    1: move left
    2: move right
    3: shoot
    """
    
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(
            self, 
            render_mode: str = None,
            max_steps: int = 12000,
            frame_skip: int = 2,
            start_level: int = 1,
            max_level: int = 4,
            img_width: int = 84,
            img_height: int = 84,
        ):
        """
        Initializes the Vision-based Space Invaders Gym environment.
        
        Args:
            render_mode: 'human' or 'rgb_array' for rendering
            max_steps: Maximum steps per episode
            frame_skip: Number of frames to skip between actions
            start_level: Initial level (1-4)
            max_level: Maximum level for curriculum
            img_width: Width of frame (default 84)
            img_height: Height of frame (default 84)
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.start_level = int(np.clip(start_level, 1, 4))
        self.max_level = int(np.clip(max_level, 1, 4))
        self.img_width = img_width
        self.img_height = img_height

        # Observation: stacked frames (84x84x4)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(self.img_height, self.img_width, 4), 
            dtype=np.uint8
        )
        
        # Action space: 4 actions
        self.action_space = gym.spaces.Discrete(4)

        self.game = Game()

        self.steps = 0
        self._last_score = 0
        self._last_lives = 3
        self._last_level_index = 1
        self._completed_game = False
        self._curriculum_completed = False
        self.boss_lives = 100
        
        # Frame buffer para stacking
        self.frame_buffer = deque(maxlen=4)
        
        self.info = {
            "score": 0,
            "lives": 3,
            "level": self.start_level,
            "completed_game": False,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.game.reset_game()
        self._set_start_level_state()
        
        self.steps = 0
        self._last_score = self.game.score
        self._last_lives = self.game.player_lives
        self._last_level_index = self._get_level_index()
        self._completed_game = False
        self._curriculum_completed = False
        self.boss_lives = 100
        
        # Inicializar frame buffer
        self.frame_buffer.clear()
        initial_frame = self._preprocess_frame(self.game.get_frame())
        for _ in range(4):
            self.frame_buffer.append(initial_frame)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: int):
        """Execute one step in the environment."""
        total_reward = 0.0
        terminated = False

        for _ in range(self.frame_skip):
            if self.game.game_over:
                terminated = True
                break

            self.steps += 1
            self._apply_action(action)

            self.game.level_manager()
            self.game.handle_collisions()
            self._update_groups()

            self._completed_game = self._is_game_completed()
            self._curriculum_completed = self._is_curriculum_target_completed()

            reward = self._compute_reward()
            total_reward += reward

            if self.game.game_over or self._completed_game or self._curriculum_completed:
                terminated = True
                break

        truncated = self.steps >= self.max_steps
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, total_reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        self._draw_state()
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            return None

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.game.screen)
            return np.transpose(frame, (1, 0, 2))

        return None

    def close(self):
        """Close the environment."""
        pygame.quit()

    def _apply_action(self, action: int):
        """Apply the action to the game state."""
        if self.game.cooldown > 0:
            self.game.cooldown -= 1

        # Actions: 0=no-op, 1=left, 2=right, 3=shoot
        move_left = action == 1
        move_right = action == 2
        shoot = action == 3

        if move_left and self.game.player.x > 0:
            self.game.player.x -= PLAYER_VEL

        if move_right and self.game.player.x < WINDOW_WIDTH - 60:
            self.game.player.x += PLAYER_VEL

        if shoot and self.game.cooldown == 0 and self.game.player_is_alive:
            self.game.bullets_group.add(
                Bullet(
                    -1,
                    (self.game.player.centerx, self.game.player.y - 15),
                    RED,
                    5, 
                    17, 
                    10,  # bullet velocity
                    0,
                )
            )
            self.game.cooldown = 40

    def _update_groups(self):
        """Update all sprite groups."""
        if self.game.level_1:
            self.game.alien_group.update(self.game, 70, 100, 3)
        if self.game.level_2:
            self.game.alien_group.update(self.game, 40, 300, 3)
        if self.game.level_3:
            self.game.alien_group.update(self.game, 60, 600, 2)

        self.game.tech_alien_group.update(self.game)
        self.game.braincell_group.update(self.game)
        self.game.bullets_group.update()
        self.game.live_group.update()
        self.game.alien_explosions_group.update()
        self.game.player_explosions_group.update()
        self.game.lives_explosions_group.update()
        self.game.teleport_group.update()
        self.game.teleport_away_group.update()

    def _compute_reward(self) -> float:
        """Compute reward based on game state."""
        score_gain = self.game.score - self._last_score
        lives_delta = self.game.player_lives - self._last_lives
        current_level = self._get_level_index()
        level_progress = current_level - self._last_level_index

        reward = 0.0
        
        # Reward for destroying enemies/gaining score
        reward += float(score_gain) * 8
        
        # Small reward for staying alive
        reward += 0.005

        # Reward for advancing levels
        if level_progress > 0:
            reward += 20.0 * float(level_progress)

        # Penalty for losing lives
        if lives_delta < 0:
            reward += float(lives_delta) * 20.0

        # Penalty for gaining lives (unexpected)
        if lives_delta > 0:
            reward -= float(lives_delta) * 15.0

        # Penalty for game over
        if self.game.game_over:
            reward -= 100.0

        # Big reward for completing the game
        if self._completed_game:
            reward += 120.0

        # Reward for boss damage
        if self.game.level_4 and len(self.game.braincell_group) > 0:
            boss_actual_lives = self.game.braincell_group.sprites()[0].lives
            boss_lives_delta = boss_actual_lives - self.boss_lives
            reward += float(-boss_lives_delta) * 10.0
            self.boss_lives = boss_actual_lives

        # Reward for curriculum completion
        if self._curriculum_completed and self.max_level < 4:
            reward += 40.0

        self._last_score = self.game.score
        self._last_lives = self.game.player_lives
        self._last_level_index = current_level

        return reward

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame: resize and convert to grayscale.
        
        Args:
            frame: Raw frame from game
            
        Returns:
            Preprocessed frame (84x84) as uint8
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        frame = cv2.resize(frame, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        
        return frame.astype(np.uint8)

    def _get_observation(self) -> np.ndarray:
        """Get the current observation (stacked frames)."""
        # Get current frame from game
        current_frame = self.game.get_frame()
        processed_frame = self._preprocess_frame(current_frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Stack frames: (84, 84, 4)
        obs = np.stack(list(self.frame_buffer), axis=-1)
        
        return obs.astype(np.uint8)

    def _draw_state(self):
        """Draw the game state."""
        
        self.game.screen.fill(BLACK)
        if not self.game.game_over:
            self.game.draw_bg_and_ui()
            if self.game.player_is_alive:
                self.game.screen.blit(self.game.player_img, (self.game.player.x, self.game.player.y))
            
            self.game.alien_group.draw(self.game.screen)
            self.game.tech_alien_group.draw(self.game.screen)
            self.game.braincell_group.draw(self.game.screen)
            
            for boss in self.game.braincell_group:
                boss.draw_health(self.game.screen, self.game.hp_img)
            
            self.game.bullets_group.draw(self.game.screen)
            self.game.live_group.draw(self.game.screen)
            self.game.alien_explosions_group.draw(self.game.screen)
            self.game.player_explosions_group.draw(self.game.screen)
            self.game.lives_explosions_group.draw(self.game.screen)
            self.game.teleport_group.draw(self.game.screen)
            self.game.teleport_away_group.draw(self.game.screen)

            # Draw level text
            level_text = None
            if self.game.level_1:
                level_text = self.game.myfont.render("Level 1", True, RED)
            elif self.game.level_2:
                level_text = self.game.myfont.render("Level 2", True, RED)
            elif self.game.level_3:
                level_text = self.game.myfont.render("Level 3", True, RED)
            elif self.game.level_4:
                level_text = self.game.myfont.render("Level 4", True, RED)

            if level_text is not None and self.game.level_timer > 0:
                self.game.level_timer -= 1
                text_rect = level_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
                self.game.screen.blit(level_text, text_rect)
        else:
            self.game.player_explosions_group.draw(self.game.screen)
            self.game.player_explosions_group.update()
            if len(self.game.player_explosions_group) == 0:
                self.game.draw_game_over()

    def _get_level_index(self) -> int:
        """Get current level index."""
        if self.game.level_1:
            return 1
        if self.game.level_2:
            return 2
        if self.game.level_3:
            return 3
        return 4

    def _get_info(self) -> dict:
        """Get info dictionary."""
        return {
            "score": self.game.score,
            "lives": self.game.player_lives,
            "step": self.steps,
            "game_over": self.game.game_over,
            "level": self._get_level_index(),
            "completed_game": self._completed_game,
            "curriculum_completed": self._curriculum_completed,
        }

    def _is_game_completed(self) -> bool:
        """Check if game is completed (boss defeated)."""
        return (
            self.game.level_4
            and self.game.times_done_level_4 > 0
            and len(self.game.braincell_group) == 0
            and not self.game.game_over
            and self.game.player_is_alive
        )

    def _is_curriculum_target_completed(self) -> bool:
        """Check if curriculum target level is completed."""
        if self.max_level >= 4:
            return False
        return self._get_level_index() > self.max_level

    def _set_start_level_state(self):
        """Set up the game state for the start level."""
        if self.start_level == 1:
            return

        self.game.level_1 = False
        self.game.level_2 = False
        self.game.level_3 = False
        self.game.level_4 = False
        self.game.alien_group.empty()
        self.game.tech_alien_group.empty()
        self.game.braincell_group.empty()
        self.game.alien_countdown = 0

        if self.start_level == 2:
            self.game.level_2 = True
            self.game.create_aliens(4, 5, 157)
            self.game.level_timer = 200
            return

        if self.start_level == 3:
            self.game.level_3 = True
            self.game.times_done_level_3 = 1
            for _ in range(6):
                self.game.tech_alien_group.add(
                    TechAlien(random.randint(0, 800), random.randint(-50, 100), 2, 50)
                )
            self.game.level_timer = 300
            return

        self.game.level_4 = True
        self.game.times_done_level_4 = 1
        self.game.braincell_group.add(Braincell(350, -150, 2))
        self.game.level_timer = 300

    def _apply_action(self, action: int):
        """Applies the given action to the game state."""

        if self.game.cooldown > 0:
            self.game.cooldown -= 1
        
        move_left = action == 0
        move_right = action == 1
        shoot = action == 2

        if move_left and self.game.player.x > 0:
            self.game.player.x -= PLAYER_VEL 

        if move_right and self.game.player.x < WINDOW_WIDTH - self.game.player.width:
            self.game.player.x += PLAYER_VEL
        
        if shoot and self.game.cooldown == 0 and self.game.player.can_shoot():
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

        
    


        





    
    

