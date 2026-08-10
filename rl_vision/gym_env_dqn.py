"""
gym_env_dqn.py  –  Entorno específico para DQN (Space Invaders).

Diferencias clave respecto a gym_env_vision.py (PPO):
  1. Penalización de estancamiento más agresiva: el agente es castigado
     de forma exponencial si pasan muchos steps sin matar un alien.
  2. Cooldown reducido a 20 (de 40): más disparos posibles → señal más densa.
  3. Bonus de disparo exitoso explícito: DQN necesita crédito inmediato
     porque su propagación de gradiente es más lenta que PPO+GAE.
  4. Reward de wave-clear aumentada a 6.0 (de 4.0).
  5. Penalización de remaining_aliens al final escalada a 0.15 (de 0.08).
  6. Sin reward de supervivencia pasiva: elimina el incentivo a quedarse
     quieto (que era el síntoma principal del video).
"""

from typing import Optional
import warnings

import gymnasium as gym
import numpy as np
from collections import deque
import cv2

import sys
import os
import random

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import pygame

# Obtener directorio actual y añadir el juego al path
current_dir = os.path.dirname(os.path.abspath(__file__))
game_src_dir = os.path.join(os.path.dirname(current_dir), "space-invaders-game", "src")
if game_src_dir not in sys.path:
    sys.path.append(game_src_dir)

from main import Game
from settings import WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_VEL, RED, BLACK, BULLET_VEL
from entities import TechAlien, Braincell
from effects import Bullet


class SpaceInvadersDQNEnv(gym.Env):
    """
    Gym environment optimizado para DQN.

    Acciones:
        0: no-op
        1: move left
        2: move right
        3: shoot
        4: move left + shoot   (sólo si enable_combo_actions=True)
        5: move right + shoot  (sólo si enable_combo_actions=True)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 60}

    # ─── Hiperparámetros de reward (fáciles de tunear) ───────────────────────
    # Escala del score del juego
    SCORE_SCALE          = 0.1

    # Bonus por kill inmediato (señal densa, crítica para DQN)
    KILL_BONUS           = 0.5

    # Bonus adicional cuando el kill fue con combo move+shoot
    COMBO_KILL_BONUS     = 0.15    # Premia disparar en movimiento

    # Bonus al limpiar la oleada completa
    WAVE_CLEAR_BONUS     = 6.0

    # Penalización de estancamiento: empieza en 0, crece hasta MAX_STAGNATION
    STAGNATION_SCALE     = 0.00008
    MAX_STAGNATION       = 0.05

    # Steps sin kill a partir de los cuales se activa la penalización
    STAGNATION_GRACE     = 30       # Margen sin penalizar tras el último kill

    # Penalización por perder vida
    LIFE_LOSS_PENALTY    = 8.0

    # Penalización por game over
    GAME_OVER_PENALTY    = 8.0

    # Bonus por completar el juego
    GAME_CLEAR_BONUS     = 10.0

    # Penalización final por aliens restantes (al truncar o terminar)
    REMAINING_ALIEN_SCALE = 0.15
    MAX_REMAINING_PENALTY = 6.0

    # Cooldown entre disparos (en steps de juego, no de env)
    SHOOT_COOLDOWN       = 20
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        render_mode: str = None,
        max_steps: int = 12000,
        frame_skip: int = 2,
        start_level: int = 1,
        max_level: int = 4,
        img_width: int = 96,
        img_height: int = 112,
        enable_combo_actions: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.start_level = int(np.clip(start_level, 1, 4))
        self.max_level = int(np.clip(max_level, 1, 4))
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.enable_combo_actions = bool(enable_combo_actions)

        if self.img_width <= 0 or self.img_height <= 0:
            raise ValueError(
                f"img_width y img_height deben ser > 0, recibido {self.img_width}x{self.img_height}"
            )

        # Observation: 4 frames apilados en canal (H, W, 4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.img_height, self.img_width, 4),
            dtype=np.uint8,
        )

        self.action_space = gym.spaces.Discrete(6 if self.enable_combo_actions else 4)

        self.game = Game()

        self.steps = 0
        self._last_score = 0
        self._last_lives = 3
        self._last_level_index = 1
        self._last_alien_count = 0
        self._steps_since_last_kill = 0
        self._last_action = 0
        self._completed_game = False
        self._curriculum_completed = False

        self.frame_buffer = deque(maxlen=4)

        self.info = {
            "score": 0,
            "lives": 3,
            "level": self.start_level,
            "completed_game": False,
        }

    # ─── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
        self._last_alien_count = len(self.game.alien_group)
        self._steps_since_last_kill = 0
        self._last_action = 0
        self._completed_game = False
        self._curriculum_completed = False

        self.frame_buffer.clear()
        self._draw_state()
        initial_frame = pygame.surfarray.array3d(self.game.screen)
        initial_frame = np.transpose(initial_frame, (1, 0, 2))
        initial_frame = self._preprocess_frame(initial_frame)
        for _ in range(4):
            self.frame_buffer.append(initial_frame)

        return self._get_observation(), self._get_info()

    def step(self, action: int):
        total_reward = 0.0
        terminated = False
        self._last_action = int(action)

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

        # Penalización final por aliens vivos (desincentiva rendirse)
        if (terminated or truncated) and len(self.game.alien_group) > 0:
            remaining = float(len(self.game.alien_group))
            total_reward -= min(self.MAX_REMAINING_PENALTY, self.REMAINING_ALIEN_SCALE * remaining)

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
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

    # ─── Internos ─────────────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        """Apply the action to the game state."""
        if self.game.cooldown > 0:
            self.game.cooldown -= 1

        if self.enable_combo_actions:
            # Actions:
            # 0=no-op, 1=left, 2=right, 3=shoot, 4=left+shoot, 5=right+shoot
            move_left = action in (1, 4)
            move_right = action in (2, 5)
            shoot = action in (3, 4, 5)
        else:
            # Classic actions: 0=no-op, 1=left, 2=right, 3=shoot
            move_left = action == 1
            move_right = action == 2
            shoot = action == 3

        if move_left and self.game.player.x > 0:
            self.game.player.x -= PLAYER_VEL

        if move_right and self.game.player.x < WINDOW_WIDTH - self.game.player.width:
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
            self.game.cooldown = self.SHOOT_COOLDOWN

    def _update_groups(self):
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
        score_gain          = self.game.score - self._last_score
        lives_delta         = self.game.player_lives - self._last_lives
        current_alien_count = len(self.game.alien_group)
        alien_kills         = self._last_alien_count - current_alien_count

        reward = 0.0

        # ── Señal de score (escala pequeña) ──────────────────────────────────
        reward += float(score_gain) * self.SCORE_SCALE

        # ── ELIMINADA la reward de supervivencia pasiva (+0.0005) ─────────────
        # Razón: incentivaba quedarse quieto sin atacar.

        # ── Señal densa de combate ────────────────────────────────────────────
        if alien_kills > 0:
            reward += self.KILL_BONUS * float(alien_kills)

            # Bonus extra si el kill fue con acción de combo (move+shoot)
            if self._last_action in (4, 5):
                reward += self.COMBO_KILL_BONUS * float(alien_kills)

            self._steps_since_last_kill = 0
        elif current_alien_count > 0:
            self._steps_since_last_kill += 1

            # Grace period: no penalizar los primeros STAGNATION_GRACE steps
            if self._steps_since_last_kill > self.STAGNATION_GRACE:
                effective_steps = self._steps_since_last_kill - self.STAGNATION_GRACE
                # Penalización progresiva con techo más alto que en PPO
                reward -= min(self.MAX_STAGNATION, self.STAGNATION_SCALE * float(effective_steps))

        # ── Bonus de wave clear ───────────────────────────────────────────────
        if current_alien_count == 0 and self._last_alien_count > 0:
            reward += self.WAVE_CLEAR_BONUS

        # ── Penalización por perder vida ──────────────────────────────────────
        if lives_delta < 0:
            reward += float(lives_delta) * self.LIFE_LOSS_PENALTY

        # ── Game over ─────────────────────────────────────────────────────────
        if self.game.game_over:
            reward -= self.GAME_OVER_PENALTY

        # ── Completar el juego ────────────────────────────────────────────────
        if self._completed_game:
            reward += self.GAME_CLEAR_BONUS

        # ── Actualizar estado interno ─────────────────────────────────────────
        self._last_score        = self.game.score
        self._last_lives        = self.game.player_lives
        self._last_level_index  = self._get_level_index()
        self._last_alien_count  = current_alien_count

        return float(np.clip(reward, -10.0, 10.0))

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def _get_observation(self) -> np.ndarray:
        self._draw_state()
        current_frame = pygame.surfarray.array3d(self.game.screen)
        current_frame = np.transpose(current_frame, (1, 0, 2))
        processed_frame = self._preprocess_frame(current_frame)
        self.frame_buffer.append(processed_frame)
        obs = np.stack(list(self.frame_buffer), axis=-1)
        return obs.astype(np.uint8)

    def _draw_state(self):
        self.game.screen.fill(BLACK)
        if not self.game.game_over:
            self.game.draw_bg_and_ui()
            if self.game.player_is_alive:
                self.game.screen.blit(
                    self.game.player_img,
                    (self.game.player.x, self.game.player.y),
                )

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
                text_rect = level_text.get_rect(
                    center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20)
                )
                self.game.screen.blit(level_text, text_rect)
        else:
            self.game.player_explosions_group.draw(self.game.screen)
            self.game.player_explosions_group.update()
            if len(self.game.player_explosions_group) == 0:
                self.game.draw_game_over()

    def _get_level_index(self) -> int:
        if self.game.level_1:
            return 1
        if self.game.level_2:
            return 2
        if self.game.level_3:
            return 3
        return 4

    def _get_info(self) -> dict:
        return {
            "score": self.game.score,
            "lives": self.game.player_lives,
            "step": self.steps,
            "game_over": self.game.game_over,
            "level": self._get_level_index(),
            "alien_count": len(self.game.alien_group),
            "completed_game": self._completed_game,
            "curriculum_completed": self._curriculum_completed,
        }

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
