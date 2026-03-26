import pygame
import os
import random
from settings import *
from effects import Bullet, TeleportAway, Teleport

class Aliens(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        if random.randint(1, 4) > 1:
            self.image = pygame.transform.scale((pygame.image.load(os.path.join(ASSETS_PATH, "sprites", "ovni.png"))), (OVNI_WIDTH, OVNI_HEIGHT))
            self.red = False
        else:
            self.image = pygame.transform.scale((pygame.image.load(os.path.join(ASSETS_PATH, "sprites", "healer_ovni.png"))), (OVNI_WIDTH, OVNI_HEIGHT))
            self.red = True
            
        self.rect = self.image.get_rect(center=[x, y])
        self.move_counter = 0
        self.move_direction = 1

    def update(self, game, distance, possibility, speed):
        self.rect.x += speed * self.move_direction
        self.move_counter += 1
        if abs(self.move_counter) > distance:
            self.rect.y += 20 # OVNI_VEL_Y
            self.move_direction *= -1
            self.move_counter *= self.move_direction
        if random.randint(3, possibility) == 3:
            bullet = Bullet(1, (self.rect.centerx, self.rect.bottom), GREEN, 5, 17, BULLET_VEL, 0)
            game.bullets_group.add(bullet)
        if self.rect.top > WINDOW_HEIGHT:
            self.kill()

class TechAlien(pygame.sprite.Sprite):
    def __init__(self, x, y, velocity, possibility):
        super().__init__()
        self.anim_list = []
        folder = os.path.join(ASSETS_PATH, "animations", "TECH_ALIEN")
        for i in range(10): # 0 a 9
            img = pygame.transform.scale(pygame.image.load(os.path.join(folder, f"pixil-frame-{i}.png")), (90, 90))
            self.anim_list.append(img)
            
        self.velocity = velocity
        self.possibility = possibility
        self.index = 0
        self.lives = 3
        self.counter = 0
        self.image = self.anim_list[self.index]
        self.rect = self.image.get_rect(center=[x, y])
        self.direction = 1
    
    def update(self, game):
        anim_speed = 3
        self.counter += 1
        
        if self.counter >= anim_speed:
            self.index = (self.index + 1) % len(self.anim_list)
            self.counter = 0
            self.image = self.anim_list[self.index] 

        if self.rect.y >= 570 or self.rect.y <= -100:
            ta = TeleportAway(self.rect.centerx, self.rect.centery)
            game.teleport_away_group.add(ta)
            self.rect.y = random.randint(-300, 400)
            self.rect.x = random.randint(-50, 450)
            t = Teleport(self.rect.centerx, self.rect.bottom, game)
            game.teleport_group.add(t)

        if random.randint(3, self.possibility) == 3:
            bullet = Bullet(1, (self.rect.centerx, self.rect.centery), BLUE, 20, 10, 8, random.randint(-4, 10))
            game.bullets_group.add(bullet)

        self.rect.y += self.velocity * self.direction

class Braincell(pygame.sprite.Sprite):
    def __init__(self, x, y, velocity):
        super().__init__()
        # Carga masiva limpia
        self.anim_base = [pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_PATH, "animations", "BRAINCELL", f"pixil-frame-{i}.png")), (400, 400)) for i in range(19)]
        self.anim_super = [pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_PATH, "animations", "BRAINCELL_2", f"pixil-frame-{i}.png")), (400, 400)) for i in range(1, 19)]
        self.anim_trans = [pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_PATH, "animations", "BRAINCELL_TRANSFORMATION", f"pixil-frame-{i}.png")), (400, 400)) for i in range(29)]

        self.velocity = velocity
        self.index = 0
        self.lives = 100
        self.counter = 0
        self.image = self.anim_base[self.index]
        self.rect = self.image.get_rect(center=[x, y])
        self.direction = 1

        self.pattern_0 = True
        self.pattern_1, self.pattern_2, self.pattern_3, self.pattern_4, self.pattern_5 = False, False, False, False, False
        self.pattern_1_counter, self.pattern_2_counter, self.pattern_4_counter = 0, 0, 0
        
        self.base_form, self.super_form, self.transformation = True, False, False

    def update(self, game):
        # Animaciones según forma
        anim_speed = 3 if self.base_form else 2
        self.counter += 1
        
        active_list = self.anim_base if self.base_form else (self.anim_trans if self.transformation else self.anim_super)
        
        if self.counter >= anim_speed:
            self.index += 1
            if self.index >= len(active_list):
                if self.transformation:
                    self.transformation = False
                    self.super_form = True
                    self.index = 0
                    active_list = self.anim_super
                else:
                    self.index = 0
            self.counter = 0
            self.image = active_list[self.index]

        # Comprobar si muere
        if self.lives <= 0:
            self.kill()
        
        # Iniciar transformación
        if self.lives <= 33 and self.base_form:
            self.base_form = False
            self.transformation = True
            self.index = 0

        # Movimientos (Patrones reducidos por simplicidad de lectura, mantienen tu lógica)
        if self.pattern_0:
            if self.rect.y < 90: self.rect.y += 1
            else: self.pattern_0 = False; self.pattern_1 = True
        elif not self.transformation:
            # Patrón 1
            if self.pattern_1:
                self.rect.x += (self.velocity if self.base_form else 10) * self.direction
                if self.rect.x >= 450 or self.rect.x <= -100:
                    self.direction *= -1
                    game.create_aliens(1, 1, 200)
                    self.pattern_1_counter += 1
                if random.randint(1, 1300 if self.base_form else 900) == 1:
                    self.pattern_1 = False; self.pattern_4 = True
                if self.pattern_1_counter >= (6 if self.base_form else 4):
                    self.pattern_1 = False; self.pattern_1_counter = 0
                    if random.randint(1, 2) == 1: self.pattern_2 = True
                    else: self.pattern_5 = True
                if random.randint(3, 60 if self.base_form else 30) == 3:
                    bullet = Bullet(1, (self.rect.x + 180, self.rect.y + 150), WHITE, 20, 20, 8, random.randint(-6, 6))
                    game.bullets_group.add(bullet)
                    
            # Patrón 2
            if self.pattern_2:
                if self.rect.y > -50: self.rect.y -= (4 if self.base_form else 10)
                else:
                    if self.rect.x < 400:
                        self.rect.x += (4 if self.base_form else 10)
                        self.pattern_2_counter += 1
                        if self.pattern_2_counter >= (30 if self.base_form else 15):
                            game.boss_create_tech_aliens(self.rect.x + 215, self.rect.y + 170, 1, 3 if self.base_form else 6, 50)
                            self.pattern_2_counter = 0
                    else:
                        self.pattern_2 = False; self.pattern_3 = True

            # Patrón 3
            if self.pattern_3:
                if self.rect.y < 100: self.rect.y += (4 if self.base_form else 10)
                else: self.pattern_3 = False; self.pattern_1 = True

            # Patrón 4
            if self.pattern_4:
                self.pattern_4_counter += 1
                if self.pattern_4_counter >= 60: self.rect.y += 20
                if self.rect.y >= 600:
                    self.rect.y = -200
                    self.pattern_4 = False; self.pattern_3 = True; self.pattern_4_counter = 0

            # Patrón 5
            if self.pattern_5:
                if self.rect.y > -50: self.rect.y -= (4 if self.base_form else 10)
                else:
                    if self.rect.x < 400:
                        self.rect.x += (4 if self.base_form else 10)
                        if random.randint(3, 5 if self.base_form else 4) == 3:
                            bullet = Bullet(1, (self.rect.x + 180, self.rect.y + 150), WHITE, 20, 20, 8, random.randint(-6, 6))
                            game.bullets_group.add(bullet)
                    else:
                        self.pattern_5 = False; self.pattern_3 = True

    def draw_health(self, surface, hp_img):
        health_bar_width = self.lives * 2
        pygame.draw.rect(surface, GREEN, (self.rect.x + 105, self.rect.y + 310, health_bar_width, 10))
        pygame.draw.rect(surface, WHITE, (self.rect.x + 105, self.rect.y + 310, 200, 12), 2)
        surface.blit(hp_img, (self.rect.x - 90, self.rect.y + 142))