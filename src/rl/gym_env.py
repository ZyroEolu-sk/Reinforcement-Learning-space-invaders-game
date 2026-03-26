import random
from typing import Optional


import sys
import os

import pygame

# Obtain the current directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/rl
parent_dir = os.path.dirname(current_dir)                # .../src

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from effects import Bullet
from entities import TechAlien, Braincell
from settings import BLACK, PLAYER_VEL, WINDOW_WIDTH, WINDOW_HEIGHT, RED, BULLET_VEL

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
    4: move left + shoot
    5: move right + shoot
    """
    # Metadata for rendering
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 4500,
        frame_skip: int = 2,
        start_level: int = 1,
        max_level: int = 4,
    ):
        """
        Initializes the Space Invaders Gym environment.
        Args:
            render_mode (str, optional): The mode to render the environment. Defaults to None.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 4500.
            frame_skip (int, optional): Number of frames to skip between actions. Defaults to 2.
            start_level (int, optional): Initial level for curriculum training. Defaults to 1.
            max_level (int, optional): Maximum level used as curriculum target. Defaults to 4.
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.start_level = int(np.clip(start_level, 1, 4))
        self.max_level = int(np.clip(max_level, 1, 4))

        self.action_space = gym.spaces.Discrete(6)  # 6 discrete actions
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
        self._last_level_index = 1
        self._completed_game = False
        self._curriculum_completed = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment to an initial state and returns an initial observation and info."""
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

        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def step(self, action: int):
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
        pygame.quit()

        
    def _apply_action(self, action: int):
        """Applies the given action to the game state."""
        if self.game.cooldown > 0:
            self.game.cooldown -= 1

        move_left = action in (1, 4)
        move_right = action in (2, 5)
        shoot = action in (3, 4, 5)

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
                    BULLET_VEL, 
                    0,
                )
            )
            self.game.cooldown = 40

    def _update_groups(self):
        """Updates all sprite groups in the game."""
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
        """Computes the reward based on the current game state."""
        
        score_gain = self.game.score - self._last_score
        lives_delta = self.game.player_lives - self._last_lives
        current_level = self._get_level_index()
        level_progress = current_level - self._last_level_index

        reward = 0.0
        reward += float(score_gain) * 6
        reward += 0.005

        if level_progress > 0:
            reward += 20.0 * float(level_progress)

        if lives_delta < 0:
            reward += float(lives_delta) * 10.0

        if self.game.game_over:
            reward -= 30.0

        if self._completed_game:
            reward += 120.0

        if self._curriculum_completed and self.max_level < 4:
            reward += 40.0

        self._last_score = self.game.score
        self._last_lives = self.game.player_lives
        self._last_level_index = current_level

        return reward

    def _get_level_index(self) -> int:
        if self.game.level_1:
            return 1
        if self.game.level_2:
            return 2
        if self.game.level_3:
            return 3
        return 4

    def _is_game_completed(self) -> bool:
        return (
            self.game.level_4
            and self.game.times_done_level_4 > 0
            and len(self.game.braincell_group) == 0
            and not self.game.game_over
            and self.game.player_is_alive
        )

    def _is_curriculum_target_completed(self) -> bool:
        if self.max_level >= 4:
            return False
        return self._get_level_index() > self.max_level

    def _set_start_level_state(self):
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


    def _get_observation(self) -> np.ndarray:
        """Constructs the observation vector from the current game state."""
        player = self.game.player

        enemies = list(self.game.alien_group.sprites()) + list(self.game.tech_alien_group.sprites()) + list(self.game.braincell_group.sprites())

        enemy_bullets = [bullet for bullet in self.game.bullets_group if bullet.direction == 1]
        player_bullets = [bullet for bullet in self.game.bullets_group if bullet.direction == -1]

        nearest_enemy_dx = 0.0
        nearest_enemy_dy = 0.0
        nearest_enemy_dist = 1.0

        if enemies:
            distances = []

            for enemy in enemies:
                dx = (enemy.rect.centerx - player.centerx) / WINDOW_WIDTH
                dy = (enemy.rect.centery - player.centery) / WINDOW_HEIGHT
                dist = np.sqrt(dx**2 + dy**2)
                distances.append((dist, dx, dy))
            distances.sort(key=lambda x: x[0])
            nearest_enemy_dist, nearest_enemy_dx, nearest_enemy_dy = distances[0]
        
        nearest_enemy_bullet_dx = 0.0
        nearest_enemy_bullet_dy = 0.0
        nearest_enemy_bullet_dist = 1.0

        if enemy_bullets:
            distances = []
            for bullet in enemy_bullets:
                dx = (bullet.rect.centerx - player.centerx) / WINDOW_WIDTH
                dy = (bullet.rect.centery - player.centery) / WINDOW_HEIGHT
                dist = (dx * dx + dy * dy) ** 0.5
                distances.append((dist, dx, dy))
            distances.sort(key=lambda t: t[0])
            nearest_enemy_bullet_dist, nearest_enemy_bullet_dx, nearest_enemy_bullet_dy = distances[0]
        
        """
        boss_hp_ratio = 0.0
        if len(self.game.braincell_group) > 0:
            boss = self.game.braincell_group.sprites()[0]
            boss_hp_ratio = np.clip(boss.health / boss.max_health, 0.0, 1.0)
        """
        obs = np.array(
            [
                (player.centerx / WINDOW_WIDTH) * 2.0 - 1.0,
                (player.centery / WINDOW_HEIGHT) * 2.0 - 1.0,
                (self.game.player_lives / 3.0) * 2.0 - 1.0,
                np.clip(self.game.score / 200.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(len(self.game.alien_group) / 30.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(len(self.game.tech_alien_group) / 10.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(len(self.game.braincell_group), 0.0, 1.0) * 2.0 - 1.0,
                np.clip(len(enemy_bullets) / 25.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(len(player_bullets) / 8.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(len(self.game.live_group) / 5.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(nearest_enemy_dx, -1.0, 1.0),
                np.clip(nearest_enemy_dy, -1.0, 1.0),
                np.clip(nearest_enemy_dist, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(nearest_enemy_bullet_dx, -1.0, 1.0),
                np.clip(nearest_enemy_bullet_dy, -1.0, 1.0),
                np.clip(nearest_enemy_bullet_dist, 0.0, 1.0) * 2.0 - 1.0,
                np.clip(self.game.cooldown / 40.0, 0.0, 1.0) * 2.0 - 1.0,
                1.0 if self.game.level_1 else -1.0,
                1.0 if self.game.level_2 else -1.0,
                1.0 if self.game.level_3 or self.game.level_4 else -1.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _draw_state(self):
        """Draws the current game state to the screen."""
        self.game.screen.fill(BLACK)
        if not self.game.game_over:
            self.game.draw_bg_and_ui()
            if self.game.player_is_alive:
                self.game.screen.blit(self.game.player_img, (self.game.player.x, self.game.player.y))
            
            self.game.alien_group.draw(self.game.screen)
            self.game.tech_alien_group.draw(self.game.screen)
            self.game.braincell_group.draw(self.game.screen)
            for boss in self.game.braincell_group:
                boss.draw_health_bar(self.game.screen, self.game.hp_img)
            
            self.game.bullets_group.draw(self.game.screen)
            self.game.live_group.draw(self.game.screen)
            self.game.alien_explosions_group.draw(self.game.screen)
            self.game.player_explosions_group.draw(self.game.screen)
            self.game.lives_explosions_group.draw(self.game.screen)
            self.game.teleport_group.draw(self.game.screen)
            self.game.teleport_away_group.draw(self.game.screen)

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

    
    def _get_info(self) -> dict:
        return {
            "score": self.game.score,
            "lives": self.game.player_lives,
            "step": self.steps,
            "game_over": self.game.game_over,
            "level": self._get_level_index(),
            "completed_game": self._completed_game,
            "curriculum_completed": self._curriculum_completed,
        }