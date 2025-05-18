# // IMPORTS \\ #
import pygame
import random
import math
import numpy as np
import os
IDLE, WANDER, CHASE, EAT, REPRODUCE = 0,1,2,3,4
P_IDLE, P_WANDER, FLEE, GRAZE, P_REPRODUCE = 0,1,2,3,4
YEAR_LENGTH = 20.0
PREDATOR_REPRO_AGE = 2.0 * YEAR_LENGTH
PREY_REPRO_AGE    = 1.2 * YEAR_LENGTH
FULL_GROWTH_AGE = 3.0 * YEAR_LENGTH
MAX_AGE = 20.0 * YEAR_LENGTH
MAX_PREDATORS = 50
MAX_PREY      = 100
import AI
import Old_NN
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import make_interp_spline

# // INITIALISING VARIABLES \\ #
pygame.init()
screen_width  = 1100
screen_height = 600
screen        = pygame.display.set_mode((screen_width, screen_height))
background = pygame.image.load("background.png").convert()
background = pygame.transform.smoothscale(background,
                                        (screen_width, screen_height))

def load_gif_frames(path):
    pil = Image.open(path)
    frames = []
    try:
        while True:
            frame = pil.convert("RGBA")
            data = frame.tobytes()
            size = frame.size
            surf = pygame.image.fromstring(data, size, "RGBA")
            frames.append(surf)
            pil.seek(pil.tell() + 1)
    except EOFError:
        pass
    return frames
gif_frames   = load_gif_frames("title_background.gif")
gif_index    = 0
gif_timer    = 0
GIF_FPS      = 60
predator_counts = []
prey_counts = []
time_steps = []
simulation_seconds = 0.0
plant_spawn_timer = 0

pygame.display.set_caption("Predator-Prey Simulation")
clock         = pygame.time.Clock()
first_names = ["Blimple", "Shleeby", "Plingus", "Florbam", "Zimble", "Bimplus", "Gleeby", "Flingle", "Pingus", "Limble", "Glimpus", "Flimble", "Shneeble", "Pimblus", "Sneebly", "Glimble", "Blimpy", "Zimble", "Flimsy", "Shlumpy", "Dingle", "Shlurp"]
last_names = ["Blim", "Plom", "Blip", "Stimp", "Pom", "Bing", "Flim", "Glim", "Zim", "Plop", "Shlurp", "Blimp", "Florp", "Glimp", "Ploob", "Shleeb", "Bloop", "Plim", "Zimble", "Flimsy", "Shlumpy"]
time_multiplier = 1.0
font            = pygame.font.SysFont(None, 24)
title_font     = pygame.font.SysFont(None, 48)
spectate_target = None
zoom_level     = 1.0
cam_offset_x    = 0
cam_offset_y    = 0
MAX_STEPS = 1000
MAX_IDLE = 60
predator_group = pygame.sprite.Group()
prey_group = pygame.sprite.Group()
predator_base_size = 12
predator_vision_distance = 100
dragging_vision_slider = False
dragging_size_slider = False

# // UNIVERSAL FUNCTIONS \\ #
def get_world_coords(mx, my):
    """Convert screen coords → world coords under current camera."""
    world_x = (mx - cam_offset_x) / zoom_level
    world_y = (my - cam_offset_y) / zoom_level
    return world_x, world_y

def handle_spectate_click(mx, my):
    """Check if we clicked a sprite; if so, set spectate_target & zoom."""
    global spectate_target, zoom_level
    wx, wy = get_world_coords(mx, my)
    for group in (predator_group, prey_group):
        for spr in group:
            if spr.rect.collidepoint(wx, wy):
                spectate_target = spr
                zoom_level      = 2.0    # or whatever you like
                return True
    return False


# // CLASSES \\ #
class predator(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        # position & appearance
        self.x = float(x)
        self.y = float(y)
        self.color = (255, 255, 255)
        variation = random.uniform(0.9, 1.1)
        self.full_size = max(1, int(predator_base_size * random.uniform(0.9, 1.1)))
        self.size     = self.full_size
        self.name  = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        
        # idle movement
        self.moving          = True
        self.steps_remaining = random.randint(60, 150)
        angle               = random.uniform(0, 2 * math.pi)
        self.angle          = angle
        self.direction_x    = math.cos(angle)
        self.direction_y    = math.sin(angle)
        self.idle_time_remaining = 0
        self.just_spawned = True
        
        # reproduction
        self.age_seconds = 0.0
        self.growth_rate = random.uniform(0.3, 1)
        self.growth_factor = 0.5
        self.fertility = random.uniform(0.9, 1)
        self.children_count = 0

        # vision & hunger
        self.fov             = math.radians(random.randint(60, 100))
        self.vision_distance = max(1, int(predator_vision_distance * variation))
        self.max_hunger      = self.size * random.randint(70, 90)
        self.hunger          = self.max_hunger
        self.energy_usage    = self.size * 0.07

        # chase speed & acceleration
        base_speed             = self.size * 0.05
        self.base_speed        = base_speed
        self.current_speed     = base_speed
        self.max_chase_speed   = base_speed * 2.5
        self.chase_acceleration= base_speed * 0.02

        # chasing state
        self.chasing     = False
        self.target_prey = None
        self.lost_counter   = 0
        self.lost_threshold = 20
        self.action     = IDLE
        
        self.time_since_eat = 0
        # sprite image
        self.image = pygame.image.load("predimg.png").convert_alpha()
        self.image = pygame.transform.smoothscale(
            self.image,
            (int(self.size*2), int(self.size*2))
        )
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.original_image = self.image
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
    def update_growth_stats(self):
        age_ratio = min(self.age_seconds / (FULL_GROWTH_AGE * self.growth_rate), 1.0)
        self.growth_factor = 0.5 + 0.5 * age_ratio  # grows from 0.5 to 1.0
        self.size = max(1, int(self.full_size * self.growth_factor))
        self.max_hunger = self.size * 80
        self.energy_usage = self.size * 0.01
        self.base_speed = Old_NN.sigmoid_derivative(Old_NN.sigmoid(self.size)) * 50 + 0.45
        self.max_chase_speed = self.base_speed * 2.5
        self.chase_acceleration = self.base_speed * 0.02
        self.vision_distance = int(predator_vision_distance * self.growth_factor)
        self.image = pygame.transform.smoothscale(self.original_image, (self.size*2, self.size*2))
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

    def draw_vision(self, screen):
        apex = (int(self.x), int(self.y))
        left_bound = (
            int(self.x + self.vision_distance * math.cos(self.angle - self.fov/2)),
            int(self.y + self.vision_distance * math.sin(self.angle - self.fov/2))
        )
        right_bound = (
            int(self.x + self.vision_distance * math.cos(self.angle + self.fov/2)),
            int(self.y + self.vision_distance * math.sin(self.angle + self.fov/2))
        )
        tmp = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        pygame.draw.polygon(tmp, (200,200,200), [apex, left_bound, right_bound], 1)
        screen.blit(tmp, (0,0))

    def detect_prey(self, prey_list):
        found = []
        for prey in prey_list:
            dx = prey.x - self.x
            dy = prey.y - self.y
            d  = math.hypot(dx, dy)
            if d <= self.vision_distance:
                ang_to = math.atan2(dy, dx)
                diff   = (ang_to - self.angle + math.pi)%(2*math.pi)-math.pi
                if abs(diff) <= self.fov/2:
                    found.append((prey, d))
        return found

    def decide_target(self, prey_group):
        visible = self.detect_prey(prey_group)
        if not visible and self.chasing and self.target_prey:
            visible = [(self.target_prey, math.hypot(self.target_prey.x-self.x,
                                                    self.target_prey.y-self.y))]
        if visible:
            cand, dist = min(visible, key=lambda x: x[1])
            ts_norm = min(self.time_since_eat, MAX_STEPS) / MAX_STEPS 
            age_norm = min(self.age_seconds, MAX_AGE) / MAX_AGE
            state = [
                dist /     self.vision_distance,
                self.hunger/self.max_hunger,
                self.current_speed/self.max_chase_speed,
                1 if visible else 0,
                self.energy_usage,
                ts_norm,
                age_norm,
                self.children_count
            ]
            action = AI.pred_q_action(state, self)
        else:
            action = IDLE

        if action == CHASE and visible:
            self.chasing     = True
            self.target_prey = cand
        elif action == IDLE:
            self.chasing     = False
            self.target_prey = None
        elif action == EAT and self.chasing:
            # let update() handle the eat when in range
            pass                
        else:
            self.chasing     = False
            self.target_prey = None
        self.action = action
    
    def reproduce(self):
        children = []
        if random.random() > self.fertility:
            return children
        count = 1
        if self.fertility > 1 and random.random() < (self.fertility - 1):
            count = 2
        for _ in range(count):
            child = predator(self.x, self.y)
            for attr in ('size', 'base_speed', 'max_hunger', 'energy_usage', 'fov',
                        'vision_distance', 'max_chase_speed', 'chase_acceleration'):
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    mutated = val * random.uniform(0.9, 1.1)
                    setattr(child, attr, int(mutated) if attr in ('size', 'steps_remaining') else mutated)
                child.fertility = self.fertility * random.uniform(0.9, 1.1)
                child.just_spawned = True
                child.hunger = child.max_hunger * (2/3)
                child.rect = child.image.get_rect(center=(child.x, child.y))
                children.append(child)
        self.action = IDLE
        self.children_count += len(children)
        return children
        


    def update(self):
        branch = "chase" if (self.chasing and self.target_prey) else \
            "reproduce" if (self.action == REPRODUCE) else \
            "wander"   # our “else” case
        print(f"[DEBUG] Predator {self.name} doing branch={branch}, action={self.action}")
        if hasattr(self, 'just_spawned') and self.just_spawned:
            self.action = WANDER
            self.moving = True
            self.steps_remaining = random.randint(30, 100)
            ang = random.uniform(0, 2 * math.pi)
            self.angle = ang
            self.direction_x = math.cos(ang)
            self.direction_y = math.sin(ang)
            del self.just_spawned
                        
        self.age_seconds += time_multiplier * (1/60)
        self.update_growth_stats()
        if self.age_seconds >= MAX_AGE:
            self.kill()
            return
        if self.hunger <= 0:
            self.kill()
            return

        if self.chasing and self.target_prey:
            dx = self.target_prey.x - self.x
            dy = self.target_prey.y - self.y
            dist = math.hypot(dx, dy)
            if dist < 15:
                self.lost_counter = 0
                # eat
                self.target_prey.kill()
                self.hunger += max(self.max_hunger, self.hunger + self.target_prey.max_hunger)
                self.chasing = False
                self.current_speed = self.base_speed
            elif dist <= self.vision_distance:
                self.lost_counter = 0
                # accelerate chase
                self.current_speed = min(
                    self.current_speed + self.chase_acceleration * time_multiplier,
                    self.max_chase_speed
                )
                self.angle       = math.atan2(dy, dx)
                self.direction_x= math.cos(self.angle)
                self.direction_y= math.sin(self.angle)
                self.x += self.direction_x * self.current_speed * time_multiplier
                self.y += self.direction_y * self.current_speed * time_multiplier
                self.hunger -= self.current_speed * self.energy_usage * 3.5 * time_multiplier
            else:
                # lost
                self.lost_counter += 1
                if self.lost_counter >= self.lost_threshold:
                    self.chasing = False
                    self.target_prey = None
                    self.current_speed = self.base_speed
        elif self.action == REPRODUCE:
            if self.age_seconds >= PREDATOR_REPRO_AGE:
                # reproduce
                self.hunger -= self.max_hunger / 3
                for child in self.reproduce():
                    predator_group.add(child)
            self.action = IDLE
            
                
        elif self.action in (WANDER, IDLE):
            # —— idle wander with pause —— #
            self.current_speed = self.base_speed
            self.hunger -= self.current_speed *self.energy_usage * time_multiplier * 2.5

            if self.moving:
                if self.steps_remaining > 0:
                    # strolling
                    self.x               += self.direction_x * self.current_speed * time_multiplier
                    self.y               += self.direction_y * self.current_speed * time_multiplier
                    self.steps_remaining -= 1 * time_multiplier
                else:
                    # enter idle
                    self.moving             = False
                    self.idle_time_remaining = random.randint(60, 90)
            else:
                # idling
                if self.idle_time_remaining > 0:
                    self.idle_time_remaining -= 1 * time_multiplier
                else:
                    # pick new wander
                    self.moving          = True
                    self.steps_remaining = random.randint(60, 150)
                    ang                  = random.uniform(0, 2 * math.pi)
                    self.angle          = ang
                    self.direction_x    = math.cos(ang)
                    self.direction_y    = math.sin(ang)


        # clamp
        self.time_since_eat += 1 * time_multiplier
        self.x = max(10, min(screen_width-10, self.x))
        self.y = max(10, min(screen_height-10, self.y))
        self.rect.center = (int(self.x), int(self.y))
        

    def custom_draw(self, surface, prey_group, show_debug, show_vision):
        deg = -math.degrees(self.angle) - 90
        rotated = pygame.transform.rotate(self.original_image, deg)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)
        if show_vision: 
            self.draw_vision(surface)
        if show_debug:
            for prey_obj, _ in self.detect_prey(prey_group):
                pygame.draw.line(surface, (255,0,0),
                                (int(self.x),int(self.y)),
                                (int(prey_obj.x),int(prey_obj.y)), 1)


class prey(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        # position & appearance
        self.x = float(x)
        self.y = float(y)
        self.color = (125,125,125)
        self.full_size  = random.randint(5,10)
        self.size      = self.full_size

        # reproduction
        self.age_seconds = 0.0
        self.growth_rate = random.uniform(0.3, 1)
        self.growth_factor = 0.5
        
        # hunger
        self.max_hunger   = self.size * 75
        self.hunger       = self.max_hunger
        self.energy_usage = self.size * 0.01

        # idle movement
        self.moving          = True
        self.steps_remaining = random.randint(60,150)
        ang = random.uniform(0,2*math.pi)
        self.direction_x    = math.cos(ang)
        self.direction_y    = math.sin(ang)
        self.idle_time_remaining = 0
        self.angle = ang

        # flight zone
        self.flightzone_radius = random.randint(75,100)
        self.flightzone_colour = (0,200,0,125)  # semi-transparent green
        
        
        self.fertility = random.uniform(0.8, 1.2)
        self.children_count = 0
        self.just_spawned = False
        

        # chase speed & acceleration (faster than predators!)
        base_speed              = self.size * 0.05 + 0.3
        self.base_speed         = base_speed
        self.current_speed      = base_speed
        self.max_chase_speed    = base_speed * 2.5     # higher top speed
        self.chase_acceleration = base_speed * 0.08  # faster acceleration
        
        # grazing
        self._graze_target = None
        self.graze_rate = self.base_speed * 0.05

        self.name = f"{random.choice(first_names)} {random.choice(last_names)}"
        # chasing state
        self._flee_target = None
        self.flee_score   = 0.0
        self.action      = IDLE
        
        self.time_since_idle = 0

        # sprite
        self.image = pygame.image.load("preyimg.png").convert_alpha()
        self.image = pygame.transform.smoothscale(
            self.image,
            (int(self.size*2), int(self.size*2))
        )
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.original_image = self.image
        self.rect = self.image.get_rect(center=(self.x, self.y))

    
    def update_growth_stats(self):
        age_ratio = min(self.age_seconds / (FULL_GROWTH_AGE * self.growth_rate), 1.0)
        self.growth_factor = 0.5 + 0.5 * age_ratio
        self.size = max(1, int(self.full_size * self.growth_factor))
        self.max_hunger = self.size * 75
        self.energy_usage = self.size * -0.01 + 0.1
        self.base_speed = self.size * 0.05 + 0.3
        self.max_chase_speed = self.base_speed * 2.5
        self.chase_acceleration = self.base_speed * 0.08
        self.graze_rate = self.base_speed * 0.05
        self.flightzone_radius = int(100 * self.growth_factor)
        self.image = pygame.transform.smoothscale(self.original_image, (self.size*2, self.size*2))
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))
        
        
    def draw_flightzone(self, surface):
        tmp = pygame.Surface((self.flightzone_radius*2, self.flightzone_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(tmp, self.flightzone_colour,
                        (self.flightzone_radius, self.flightzone_radius),
                        self.flightzone_radius, 1)
        surface.blit(tmp, (self.x - self.flightzone_radius, self.y - self.flightzone_radius))

    def detect_predators(self, preds):
        found = []
        for p in preds:
            dx = p.x - self.x
            dy = p.y - self.y
            d  = math.hypot(dx, dy)
            if d <= self.flightzone_radius:
                found.append((p,d))
        return found
    
    def _perform_graze(self):
        # pick a plant if needed
        if (self._graze_target is None 
            or not self._graze_target.alive() 
            or self._graze_target.size <= 1):
            # look at all plants in the world
            candidates = []
            for pl in plant_group:
                dx = pl.rect.centerx - self.x
                dy = pl.rect.centery  - self.y
                if math.hypot(dx,dy) <= self.flightzone_radius:
                    candidates.append(pl)
            if candidates:
                self._graze_target = max(candidates, key=lambda pl: pl.size)
            else:
                # no plants → go wander
                self.action = P_WANDER
                return
        dx = self._graze_target.rect.centerx - self.x
        dy = self._graze_target.rect.centery  - self.y
        dist = math.hypot(dx, dy)
        
        if dist > self.size:
            self.current_speed = self.base_speed
            # step forwards at base speed
            angle = math.atan2(dy, dx)
            self.x += math.cos(angle) * self.current_speed * time_multiplier
            self.y += math.sin(angle) * self.current_speed * time_multiplier
            self.angle = angle
            # movement costs hunger
            self.hunger = max(0, self.hunger - self.current_speed * 0.25 * time_multiplier)
        else:
            # 3) Once we're on the plant, nibble as before
            eat_amt = self.graze_rate * time_multiplier
            pl      = self._graze_target
            pl.size = max(1, pl.size - eat_amt)
            pl._rebuild_image(pl.rect.centerx, pl.rect.centery)
            self.hunger = min(self.max_hunger, self.hunger + (eat_amt * 10))
        
            if pl.size <= 1:
                pl.kill()
                self._graze_target = None
        

    def decidebehaviour(self, predator_group):
        # 1) Find all predators chasing *you*
        chasing_preds = [
            p for p in predator_group
            if p.chasing and p.target_prey is self
        ]
        if chasing_preds:
            # choose nearest for distance/zone checks
            nearest = min(
                chasing_preds,
                key=lambda p: math.hypot(p.x-self.x, p.y-self.y)
            )
            dist    = math.hypot(nearest.x-self.x, nearest.y-self.y)
            in_zone = dist <= self.flightzone_radius
        else:
            nearest, dist, in_zone = None, self.flightzone_radius, False

        # 2) **Continue** fleeing if already in flee mode and predator still chasing
        if self._flee_target \
            and self._flee_target.chasing \
            and self._flee_target.target_prey is self:
            self.action = FLEE
            return

        # 3) **Start** fleeing only when predator enters your flight zone
        if chasing_preds and in_zone:
            self._flee_target = nearest
            self.flee_score   = 1.0
            self.action       = FLEE
            return

        # 4) No active flee → build state and let Q-policy decide
        #    (including true plant distances)
        plant_ds = []
        for pl in plant_group:
            dx = pl.rect.centerx - self.x
            dy = pl.rect.centery  - self.y
            d  = math.hypot(dx, dy)
            if d <= self.flightzone_radius:
                plant_ds.append(d)
        plant_norm = (min(plant_ds)/self.flightzone_radius) if plant_ds else 1.0
        age_norm = min(self.age_seconds, MAX_AGE) / MAX_AGE
        state = [
            min(dist, self.flightzone_radius) / self.flightzone_radius,
            1 - (self.hunger / self.max_hunger),
            self.current_speed / self.max_chase_speed,
            1 if in_zone and chasing_preds else 0,
            plant_norm,
            min(self.time_since_idle, MAX_IDLE) / MAX_IDLE,
            age_norm,
            self.children_count
        ]
        action = AI.prey_q_action(state, self)

        # 5) If Q-policy says anything but FLEE, clear out the flee target
        if action != FLEE:
            self._flee_target = None
            self.flee_score   = 0.0
        self.action = action
    
    def reproduce(self):
        children = []
        if random.random() > self.fertility:
            return children
        count = 1
        if self.fertility > 1 and random.random() < (self.fertility - 1):
            count = 2
        for _ in range(count):
            child = prey(self.x, self.y)
            for attr in ('size', 'base_speed', 'max_hunger', 'energy_usage', 'flight_zone_radius', 'max_chase_speed', 'chase_acceleration'):
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    mutated = val * random.uniform(0.9, 1.1)
                    setattr(child, attr, int(mutated) if attr in ('size', 'steps_remaining') else mutated)

            child.fertility = self.fertility * random.uniform(0.9, 1.1)
            child.just_spawned = True
            child.hunger = child.max_hunger * (3/5)
            child.rect = child.image.get_rect(center=(child.x, child.y))
            children.append(child)
        self.action = IDLE
        self.children_count += len(children)
        return children

    def update(self):
        branch = "Flee" if (self._flee_target and self.flee_score > 0.5) else \
            "reproduce" if (self.action == P_REPRODUCE) else \
            "wander"   # our “else” case
        print(f"[DEBUG] Prey {self.name} doing branch={branch}, action={self.action}")
        # starvation
        if hasattr(self, 'just_spawned') and self.just_spawned:
            self.action = WANDER
            self.moving = True
            self.steps_remaining = random.randint(30, 100)
            ang = random.uniform(0, 2 * math.pi)
            self.angle = ang
            self.direction_x = math.cos(ang)
            self.direction_y = math.sin(ang)
            del self.just_spawned
        self.age_seconds += time_multiplier * (1/60) 
        self.update_growth_stats()
        if self.age_seconds >= MAX_AGE:
            self.kill()
            return
        if self.hunger <= 0:
            self.kill()
            return
        
        if self._flee_target is not None and not (
            self._flee_target.chasing and self._flee_target.target_prey is self
        ):
            self._flee_target = None
            self.flee_score   = 0.0
            self.flightzone_colour = (0,200,0,125)

        # 2) otherwise, if there’s a predator in my flight-zone that’s actively chasing me, start (or keep) fleeing
        if self._flee_target is None:
            for p, d in self.detect_predators(predator_group):
                if p.chasing and p.target_prey is self:
                    # start fleeing
                    self._flee_target = p
                    self.flee_score   = 1.0
                    self.action       = FLEE
                    break
        
        # fleeing
        if self._flee_target:
            self._graze_target = None
            self.hunger = max(0, self.hunger - self.current_speed * time_multiplier * 0.4)
            # accelerate and run opposite the predator
            self.current_speed = min(
                self.current_speed + self.chase_acceleration * time_multiplier,
                self.max_chase_speed
            )
            # 1) predator-avoidance vector
            dx_pred = self.x - self._flee_target.x
            dy_pred = self.y - self._flee_target.y

            # 2) wall-avoidance vector pointing to map centre
            center_x = screen_width  * 0.5
            center_y = screen_height * 0.5
            dx_ctr  = center_x - self.x
            dy_ctr  = center_y - self.y

            # 3) blend them (beta controls how strongly they head towards centre)
            beta = 0.15
            dx = dx_pred + beta * dx_ctr
            dy = dy_pred + beta * dy_ctr

            # update angle & position
            ang = math.atan2(dy, dx)
            self.angle = ang
            self.x += math.cos(ang) * self.current_speed * time_multiplier
            self.y += math.sin(ang) * self.current_speed * time_multiplier
            self.flightzone_colour = (255,0,0,125)

        # grazing
        elif self.action == GRAZE:
            self.flightzone_colour = (0,0,125)
            self._perform_graze()
            
        elif self.action == P_REPRODUCE:
            if self.age_seconds >= PREY_REPRO_AGE:
                self.flightzone_colour = (255,192,204)
                self.hunger -= self.max_hunger / 3
                for child in self.reproduce():
                    prey_group.add(child)
            self.action = IDLE
            
        elif self.action == P_REPRODUCE and self.age_seconds < PREY_REPRO_AGE:
            self.flightzone_colour = (0,200,0,125)
            return

        # idle wander
        else:
            self.current_speed = self.base_speed
            self.hunger = max(0, self.hunger - self.current_speed * time_multiplier * 0.25)
            self.flightzone_colour = (0,200,0, 125)  # green
            speed = self.base_speed
            if self.moving:
                if self.steps_remaining > 0:
                    self.x               += self.direction_x * speed * time_multiplier
                    self.y               += self.direction_y * speed * time_multiplier
                    self.steps_remaining -= 1 * time_multiplier
                else:
                    self.moving             = False
                    self.idle_time_remaining = random.randint(30, 60)
            else:
                if self.idle_time_remaining > 0:
                    self.idle_time_remaining -= 1 * time_multiplier
                else:
                    self.moving          = True
                    self.steps_remaining = random.randint(100, 200)
                    ang                  = random.uniform(0, 2 * math.pi)
                    self.angle          = ang
                    self.direction_x    = math.cos(ang)
                    self.direction_y    = math.sin(ang)

        # clamp to screen bounds
        self.time_since_idle += 1 * time_multiplier
        self.x = max(10, min(screen_width-10, self.x))
        self.y = max(10, min(screen_height-10, self.y))
        self.rect.center = (int(self.x), int(self.y))

    def custom_draw(self, surface, show_flightzone):
        deg = -math.degrees(self.angle) - 90
        rotated = pygame.transform.rotate(self.original_image, deg)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)
        if show_flightzone:
            self.draw_flightzone(surface)


class plant(pygame.sprite.Sprite):
    def __init__(self, x, y, *groups):
        super().__init__(*groups)
        self.size        = float(random.randint(2,4))
        self.growth_rate = 0.0025
        self.max_size    = self.size * random.uniform(1.5, 2.5)
        self._rebuild_image(x, y)

    def _rebuild_image(self, cx, cy):
        s = max(1, int(self.size))
        self.image = pygame.Surface((s*2, s*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (0,200,0), (s,s), s)
        self.rect = self.image.get_rect(center=(cx,cy))

    def update(self):
        if self.size < self.max_size:
            self.size = min(self.max_size,
                            self.size + self.growth_rate * time_multiplier)
            cx, cy = self.rect.center
            self._rebuild_image(cx, cy)


# // UI FUNCTIONS \\ #
def draw_population_graph(surface, x, y, w, h):
    if len(time_steps) < 4:
        return

    font_small = pygame.font.SysFont(None, 18)

    # Compute game time (adjusted for time_multiplier)
    total_game_time = [t * time_multiplier / 20.0 for t in time_steps]  # seconds to years
    max_time = max(total_game_time)
    max_pop = max(max(predator_counts, default=1), max(prey_counts, default=1))

    # Border
    pygame.draw.rect(surface, (255, 255, 255), (x, y, w, h), 1)

    # --- Axis Ticks ---
    for i in range(0, max(int(max_pop)+1, 1), max(1, int(max_pop) // 5)):
        y_pos = y + h - int((i / max_pop) * h)
        pygame.draw.line(surface, (100, 100, 100), (x-5, y_pos), (x+5, y_pos), 1)
        label = font_small.render(str(i), True, (255, 255, 255))
        surface.blit(label, (x - 40, y_pos - 8))

    for i in range(0, int(max_time)+1, max(1, int(max_time) // 5)):
        x_pos = x + int((i / max_time) * w)
        pygame.draw.line(surface, (100, 100, 100), (x_pos, y+h-5), (x_pos, y+h+5), 1)
        label = font_small.render(f"{i}", True, (255, 255, 255))
        surface.blit(label, (x_pos - 10, y + h + 5))

    surface.blit(font_small.render("Population", True, (255, 255, 255)), (x - 60, y + h//2 - 20))
    surface.blit(font_small.render("Game Time (Years)", True, (255, 255, 255)), (x + w//2 - 50, y + h + 25))

    # --- Smooth Curve Drawing ---
    def draw_smooth_curve(data, color):
        x_vals = np.array(total_game_time)
        y_vals = np.array(data)
        if len(x_vals) < 4:
            return  # need at least 4 points for spline

        # Interpolation
        x_new = np.linspace(x_vals.min(), x_vals.max(), 300)
        spline = make_interp_spline(x_vals, y_vals, k=3)
        y_smooth = spline(x_new)

        points = [
            (x + int((t / max_time) * w), y + h - int((v / max_pop) * h))
            for t, v in zip(x_new, y_smooth)
        ]
        pygame.draw.lines(surface, color, False, points, 2)

    draw_smooth_curve(predator_counts, (255, 0, 0))
    draw_smooth_curve(prey_counts, (0, 255, 0))

def create_game_ui():
    # time controls & settings button…
    pygame.draw.rect(screen, (50,50,50), slow_down_button)
    pygame.draw.rect(screen, (255,255,255), slow_down_button, 2)
    txt = font.render("<<<", True, (255,255,255))
    screen.blit(txt, (slow_down_button.x + (slow_down_button.width - txt.get_width())//2,
                    slow_down_button.y + (slow_down_button.height - txt.get_height())//2))

    pygame.draw.rect(screen, (50,50,50), speed_up_button)
    pygame.draw.rect(screen, (255,255,255), speed_up_button, 2)
    txt = font.render(">>>", True, (255,255,255))
    screen.blit(txt, (speed_up_button.x + (speed_up_button.width - txt.get_width())//2,
                    speed_up_button.y + (speed_up_button.height - txt.get_height())//2))

    pygame.draw.rect(screen, (50,50,50), settings_button)
    pygame.draw.rect(screen, (255,255,255), settings_button, 2)
    txt = font.render("Settings", True, (255,255,255))
    screen.blit(txt, (settings_button.x + (settings_button.width - txt.get_width())//2,
                    settings_button.y + (settings_button.height - txt.get_height())//2))
    txt = font.render("FPS: " + str(int(clock.get_fps())), True, (255,255,255))
    screen.blit(txt, (10, screen_height-80))
    txt = font.render("Predators: " + str(len(predator_group)), True, (255,255,255))
    screen.blit(txt, (10, screen_height-60))
    txt = font.render("Prey: " + str(len(prey_group)), True, (255,255,255))
    screen.blit(txt, (10, screen_height-40))
    txt = font.render("Plants: " + str(len(plant_group)), True, (255,255,255))
    screen.blit(txt, (10, screen_height-20))

    tm = font.render(f"Time x{time_multiplier:.1f}", True, (255,255,0))
    screen.blit(tm, (screen_width-160, screen_height-80))
    


def create_settings_ui():
    screen.fill((0,0,0,180))
    pygame.draw.rect(screen, (50,50,50), back_button)
    pygame.draw.rect(screen, (255,255,255), back_button, 2)
    txt = font.render("Back", True, (255,255,255))
    screen.blit(txt, (back_button.x + (back_button.width - txt.get_width())//2,
                    back_button.y + (back_button.height - txt.get_height())//2))
    
    pygame.draw.rect(screen, (50,50,50), restart_button)
    pygame.draw.rect(screen, (255,255,255), restart_button, 2)
    txt = font.render("Restart", True, (255,255,255))
    screen.blit(txt, (restart_button.x + (restart_button.width - txt.get_width())//2,
                    restart_button.y + (restart_button.height - txt.get_height())//2))

    for opt in settings_checkboxes:
        box = opt["rect"]
        pygame.draw.rect(screen, (200,200,200), box)
        pygame.draw.rect(screen, (255,255,255), box, 2)
        if opt["checked"]:
            inner = box.inflate(-6, -6)
            pygame.draw.rect(screen, (0,200,0), inner)
        lbl = font.render(opt["label"], True, (255,255,255))
        screen.blit(lbl, (box.right+10, box.y+(box.height-lbl.get_height())//2))
        
def blit_world_with_camera():
    """Draw the world to a temp surface, scale & blit it under camera."""
    global cam_offset_x, cam_offset_y
    # 1) draw into world_surf
    world_surf = background.copy()
    plant_group.draw(world_surf)
    for pr in prey_group:
        pr.custom_draw(world_surf, show_flightzone)
    for p in predator_group:
        p.custom_draw(world_surf, prey_group, show_debug, show_vision)

    # 2) centre on spectate_target
    if spectate_target:
        cx, cy = spectate_target.x, spectate_target.y
        cam_offset_x = screen_width/2  - cx * zoom_level
        cam_offset_y = screen_height/2 - cy * zoom_level
    else:
        cam_offset_x = cam_offset_y = 0

    # 3) scale & blit
    scaled = pygame.transform.smoothscale(
        world_surf,
        (int(screen_width*zoom_level), int(screen_height*zoom_level))
    )
    
    #clamp
    scaled_w = int(screen_width * zoom_level)
    scaled_h = int(screen_height * zoom_level)
    
    cam_offset_x = max(min(cam_offset_x, 0), screen_width  - scaled_w)
    cam_offset_y = max(min(cam_offset_y, 0), screen_height - scaled_h)
    
    cam_offset_x = int(cam_offset_x)
    cam_offset_y = int(cam_offset_y)
    
    screen.blit(scaled, (cam_offset_x, cam_offset_y))

    # 4) scale & blit
    scaled = pygame.transform.smoothscale(
        world_surf, 
        (int(screen_width * zoom_level),
         int(screen_height * zoom_level))
    )
    screen.blit(scaled, (cam_offset_x, cam_offset_y))
    
def draw_spectate_panel():
    panel = pygame.Surface((200,100), pygame.SRCALPHA)
    panel.fill((0,0,0,180))
    screen.blit(panel, (10,10))
    screen.blit(font.render(f"Hunger: {spectate_target.hunger:.0f}/{spectate_target.max_hunger}", True, (255,255,255)), (20,20))
    screen.blit(font.render(f"Base Speed: {round(spectate_target.base_speed, 3)}", True, (255,255,255)), (20,40))
    screen.blit(font.render(f"Current Speed: {round(spectate_target.current_speed,3)}", True, (255,255,255)), (20,60))
    screen.blit(font.render(f"Current Action: {spectate_target.action}", True, (255,255,255)), (20,80))
    screen.blit(font.render(f"Current Age: {round((spectate_target.age_seconds / YEAR_LENGTH), 1)}", True, (255,255,255)), (20,100))
    screen.blit(font.render(f"Pos: ({spectate_target.x:.1f},{spectate_target.y:.1f})", True, (200,200,200)), (20,120))
    screen.blit(font.render(f"Currently Spectating: {spectate_target.name}", True, (255,255,255)), (400,40))

def create_title_ui():
    title = title_font.render("Predator / Prey Simulation", True, (255,255,255))
    screen.blit(title, (screen_width//2 - title.get_width()//2, screen_height//2 - title.get_height()//2))
    pygame.draw.rect(screen, (50,50,50), start_button)
    pygame.draw.rect(screen, (255,255,255), start_button, 2)
    txt = font.render("Start", True, (255,255,255))
    screen.blit(txt, (start_button.x + (start_button.width - txt.get_width())//2,
                    start_button.y + (start_button.height - txt.get_height())//2))

def spawn_initial_sprites():
    for _ in range(10):
        predator_group.add(predator(random.randint(10,screen_width-10),
                                random.randint(10,screen_height-10)))
        prey_group    .add(prey(   random.randint(10,screen_width-10),
                                random.randint(10,screen_height-10)))
    for _ in range(50):
        plant_group.add(plant(random.randint(10,screen_width-10),
                        random.randint(10,screen_height-10)))

def create_endscreen_ui():
    screen.fill((0,0,0))
    title = title_font.render("Simulation Over", True, (255,255,255))
    screen.blit(title, (screen_width//2 - title.get_width()//2, 100 - title.get_height()//2))
    draw_population_graph(screen, x=275, y=150, w=550, h=250)
    start_button.x = screen_width//2 - 50
    start_button.y = screen_height - 100
    pygame.draw.rect(screen, (50,50,50), start_button)
    pygame.draw.rect(screen, (255,255,255), start_button, 2)
    txt = font.render("Restart", True, (255,255,255))
    screen.blit(txt, (start_button.x + (start_button.width - txt.get_width())//2,
                    start_button.y + (start_button.height - txt.get_height())//2))
    
def create_pregame_ui():
    screen.fill((0,0,0))
    pygame.draw.rect(screen, (50,50,50), start_button)
    pygame.draw.rect(screen, (255,255,255), start_button, 2)
    txt = font.render("Start", True, (255,255,255))
    screen.blit(txt, (start_button.x + (start_button.width - txt.get_width())//2,
                    start_button.y + (start_button.height - txt.get_height())//2))
    # Size slider
    size_slider_x = 300
    size_slider_y = 300
    size_slider_w = 400
    size_slider_h = 8
    pygame.draw.rect(screen, (200,200,200), (size_slider_x, size_slider_y, size_slider_w, size_slider_h))
    size_slider_percent = (predator_base_size - 7) / 8
    size_handle_x = size_slider_x + int(size_slider_percent * size_slider_w)
    pygame.draw.circle(screen, (255,255,0), (size_handle_x, size_slider_y + size_slider_h//2), 10)
    
    txt = font.render(f"Predator Size: {predator_base_size}", True, (255,255,255))
    screen.blit(txt, (size_slider_x, size_slider_y - 30))
    
    #Vision distance slider
    vision_slider_x = 300
    vision_slider_y = 220
    vision_slider_w = 400
    vision_slider_h = 8
    pygame.draw.rect(screen, (200,200,200), (vision_slider_x, vision_slider_y, vision_slider_w, vision_slider_h))
    vision_slider_percent = (predator_vision_distance - 100) / 80
    vision_handle_x = vision_slider_x + int(vision_slider_percent * vision_slider_w)
    pygame.draw.circle(screen, (255,255,0), (vision_handle_x, vision_slider_y + vision_slider_h//2), 10)
    
    txt = font.render(f"Predator VIsion Distance: {predator_vision_distance}", True, (255,255,255))
    screen.blit(txt, (vision_slider_x, vision_slider_y - 30))
    pass




# // SPRITE GROUPS \\ #
predator_group = pygame.sprite.Group()
prey_group     = pygame.sprite.Group()
plant_group    = pygame.sprite.Group()

# // UI ELEMENTS \\ #
slow_down_button = pygame.Rect(screen_width-200, screen_height-50, 90, 40)
start_button = pygame.Rect(screen_width//2 - 50, screen_height - 100, 100, 40)
speed_up_button  = pygame.Rect(screen_width-100, screen_height-50, 90, 40)
settings_button  = pygame.Rect(screen_width-100, 10, 90, 40)
settings_button_title  = pygame.Rect(screen_width-100, 10, 90, 40)
back_button      = pygame.Rect(screen_width-100, 10, 90, 40)
restart_button = pygame.Rect(screen_width-100, screen_height-100, 90, 40)
settings_checkboxes = [
    {"label":"Show Vision Cone", "rect":pygame.Rect(100,150,20,20), "checked":False},
    {"label":"Enable Chase Lines","rect":pygame.Rect(100,200,20,20), "checked":True},
    {"label":"Show Flight Zone","rect":pygame.Rect(100,250,20,20), "checked":True},
]


decision_interval   = 20
frame_counter       = 0

# // MAIN LOOP \\ #
def run_game():
    global time_multiplier, spectate_target, zoom_level, show_debug, show_vision, show_flightzone, predator_base_size, predator_vision_distance, dragging_size_slider, gif_timer, gif_index, dragging_vision_slider, predator_counts, prey_counts, time_steps, plant_spawn_timer

    # now define your own counter for this run
    frame_counter = 0
    running       = True
    currscreen    = 'title'
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if currscreen == 'gamescreen':
                    if event.key == pygame.K_ESCAPE:
                        spectate_target = None
                        zoom_level = 1.0
                    elif event.key == pygame.K_LEFT:
                        time_multiplier = max(0.5, time_multiplier/2)
                    elif event.key == pygame.K_RIGHT:
                        time_multiplier = min(16, time_multiplier*2)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = pygame.mouse.get_pos()
                if currscreen == 'gamescreen':
                    if handle_spectate_click(mx, my):
                        continue   
                    if slow_down_button.collidepoint((mx,my)):
                        time_multiplier = max(0.5, time_multiplier/2)
                    elif speed_up_button.collidepoint((mx,my)):
                        time_multiplier = min(16, time_multiplier*2)
                    elif settings_button.collidepoint((mx,my)):
                        currscreen = 'settings'
                elif currscreen == 'settings':
                    if back_button.collidepoint((mx, my)):
                        currscreen = 'gamescreen'
                    for opt in settings_checkboxes:
                        if opt["rect"].collidepoint((mx, my)):
                            opt["checked"] = not opt["checked"]
                        if restart_button.collidepoint((mx, my)):
                            currscreen = 'endscreen'
                            prey_group.empty()
                            predator_group.empty()
                            plant_group.empty()
                            screen.fill((0,0,0))
                elif currscreen == 'title':
                    if start_button.collidepoint((mx, my)):
                        currscreen = 'pregame'
                elif currscreen == "pregame":
                    size_slider_rect = pygame.Rect(300, 300, 400, 20)
                    vision_slider_rect = pygame.Rect(300, 220, 400, 20)
                    # slightly taller to make grabbing easier
                    if size_slider_rect.collidepoint(mx, my):
                        dragging_size_slider = True
                    elif vision_slider_rect.collidepoint(mx, my):
                        dragging_vision_slider = True
                    elif start_button.collidepoint((mx, my)):
                        currscreen = 'gamescreen'
                        predator_counts = []
                        prey_counts = []
                        time_steps = []
                        simulation_seconds = 0
                        spawn_initial_sprites()
                elif currscreen == 'endscreen':
                    if start_button.collidepoint((mx, my)):
                        currscreen = 'pregame'
                        prey_group.empty()
                        predator_group.empty()
                        plant_group.empty()
                        screen.fill((0,0,0))
            elif event.type == pygame.MOUSEBUTTONUP:
                if currscreen == 'pregame':
                    dragging_vision_slider = False
                    dragging_size_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
                if currscreen == 'pregame':
                    if dragging_size_slider:
                        mx, my = pygame.mouse.get_pos()
                        rel_x = max(0, min(mx - 300, 400))
                        percent = rel_x / 400
                        predator_base_size = int(round(7 + percent * 8))
                    elif dragging_vision_slider:
                        mx, my = pygame.mouse.get_pos()
                        rel_x = max(0, min(mx - 300, 400))
                        percent = rel_x / 400
                        predator_vision_distance = int(round(100 + percent * 80))

        clock.tick(60)

        if currscreen=='gamescreen':
            screen.blit(background, (0,0))
            if len(predator_group) == 0:
                currscreen = 'endscreen'
            if len(prey_group) == 0:
                currscreen = 'endscreen'
            # spawn plants periodically
            plant_spawn_timer += (1/60) * time_multiplier
            if plant_spawn_timer >= 4.0:  # spawn every 4 real seconds, scaled by time_multiplier
                plant_spawn_timer = 0.0
                for _ in range(1):
                    plant_group.add(plant(random.randint(10, screen_width - 10),
                                        random.randint(10, screen_height - 10)))

            # AI decisions once per interval
            if frame_counter % decision_interval == 0:
                for pred in predator_group:
                    pred.decide_target(prey_group)
                for pr in prey_group:
                    pr.decidebehaviour(predator_group)
            if frame_counter % 10 == 0:
                predator_counts.append(len(predator_group))
                prey_counts.append(len(prey_group))
                time_steps.append(simulation_seconds/20)

            # updates
            predator_group.update()
            prey_group.update()
            plant_group.update()

            # draws
            show_vision = next(opt for opt in settings_checkboxes
                            if opt["label"]=="Show Vision Cone")["checked"]
            show_debug  = next(opt for opt in settings_checkboxes
                            if opt["label"]=="Enable Chase Lines")["checked"]
            show_flightzone = next(opt for opt in settings_checkboxes
                            if opt["label"]=="Show Flight Zone")["checked"]
            blit_world_with_camera()
            create_game_ui()
            if spectate_target:
                draw_spectate_panel()
            simulation_seconds += (1/60) * time_multiplier

        elif currscreen == 'settings':
            create_settings_ui()
        
        elif currscreen == 'title':
            create_title_ui()
        
        elif currscreen == 'pregame':
            create_pregame_ui()
            
        elif currscreen == 'endscreen':
            create_endscreen_ui()      

        pygame.display.flip()
        frame_counter += 1

if __name__ == "__main__":
    run_game()
