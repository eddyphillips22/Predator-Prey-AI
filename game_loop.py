# // IMPORTS \\ #
import pygame
import random
import math
import numpy as np
import os
import csv

SAVE_FILE_PATH = "saves.csv"
IDLE, WANDER, CHASE, EAT, REPRODUCE = 0,1,2,3,4
P_IDLE, P_WANDER, FLEE, GRAZE, P_REPRODUCE = 0,1,2,3,4
YEAR_LENGTH = 20.0
PREDATOR_REPRO_AGE = 2.0 * YEAR_LENGTH
PREY_REPRO_AGE    = 1.2 * YEAR_LENGTH
FULL_GROWTH_AGE = 3.0 * YEAR_LENGTH
MAX_AGE = 20.0 * YEAR_LENGTH
MAX_PREDATORS = 50
MAX_PREY      = 100
PREDATOR_EAT_RANGE = 15
VISION_UNLOCK_XP = 20
FERTILITY_UNLOCK_XP = 150
DEFAULT_SLOT = {
    "used":       False,
    "XP":         0,
    "size_min":   10,
    "size_max":  13,
    "vision_min":110,
    "vision_max":140,
    "fertility_min": 0.8,
    "fertility_max": 1,
}
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
background = pygame.image.load("assets/background.png").convert()
background_computer = pygame.image.load("assets/computer.png").convert()
background_computer = pygame.transform.scale(
                    background_computer,
                    (screen_width * 1.3, screen_height * 1.3)
                )
background = pygame.transform.smoothscale(background,
                                        (screen_width, screen_height))
active_save_slot = None

from PIL import Image, ImageSequence
import pygame

def load_gif_frames(path):

    pil = Image.open(path)
    print(f"[DEBUG] loading {path} → is_animated: {getattr(pil, 'is_animated', False)}, n_frames: {getattr(pil, 'n_frames', 1)}")
    print(f"[DEBUG] pil.info keys: {pil.info}") 
    frames = []
    durations = []

    # how many frames in this GIF?
    try:
        total = pil.n_frames
    except AttributeError:
        total = 1

    # keep a full RGBA canvas to composite “delta” frames
    base = Image.new("RGBA", pil.size)

    for i in range(total):
        pil.seek(i)

        # some GIFs only store the changed region – paste into base
        frame = pil.convert("RGBA")
        base.paste(frame, (0, 0), frame)

        # convert to pygame Surface
        surf = pygame.image.fromstring(base.tobytes(), base.size, "RGBA")
        frames.append(surf)

        # get this frame’s duration (ms), fallback to 100ms
        dur_ms = pil.info.get("duration", 100)
        durations.append(max(dur_ms, 20) / 1000.0)

    return frames, durations

def play_gif_sequence(paths):
    """
    Play each GIF (given by its file‐path) exactly once,
    honouring each frame’s own duration, then return.
    """
    clock = pygame.time.Clock()

    for path in paths:
        frames, durations = load_gif_frames(path)
        for frame_surf, dur in zip(frames, durations):
            # how long (in ms) this frame should stay on screen
            target_time = pygame.time.get_ticks() + int(dur * 1000)

            while pygame.time.get_ticks() < target_time:
                # 1) handle quit events so window remains responsive
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        return

                # 2) draw this frame
                screen.fill((0,0,0))
                frame_surf = pygame.transform.scale(
                    frame_surf,
                    (476, 269)
                )
                screen.blit (background_computer, (screen_width // 2 - background_computer.get_width() // 2, screen_height // 2 - background_computer.get_height() // 2 + 100))
                screen.blit(frame_surf, (screen_width // 2 - frame_surf.get_width() // 2 - 7, screen_height // 2 - frame_surf.get_height() // 2 - 72))
                pygame.display.flip()

                # 3) tick at 60fps so we don't peg the CPU
                clock.tick(60)

gif_index    = 0
gif_timer    = 0
GIF_FPS      = 60
predator_counts = []
prey_counts = []
time_steps = []
simulation_seconds = 0.0
plant_spawn_timer = 0
session_size_min = None
session_size_max = None
session_vision_min = None
session_vision_max = None
session_fertility_min = None
session_fertility_max = None
pygame.display.set_caption("Predator-Prey Simulation")
clock         = pygame.time.Clock()
first_names = ["Blimple", "Shleeby", "Plingus", "Florbam", "Zimble", "Bimplus", "Gleeby", "Flingle", "Pingus", "Limble", "Glimpus", "Flimble", "Shneeble", "Pimblus", "Sneebly", "Glimble", "Blimpy", "Zimble", "Flimsy", "Shlumpy", "Dingle", "Shlurp"]
last_names = ["Blim", "Plom", "Blip", "Stimp", "Pom", "Bing", "Flim", "Glim", "Zim", "Plop", "Shlurp", "Blimp", "Florp", "Glimp", "Ploob", "Shleeb", "Bloop", "Plim", "Zimble", "Flimsy", "Shlumpy"]
time_multiplier = 1.0
font            = pygame.font.SysFont(None, 24)
title_font     = pygame.font.SysFont(None, 48)
spectate_target = None
player_controlled = None
controlling_mode = False
move_x       = 0
move_y       = 0
sprinting    = False
trying_eat     = False
trying_reproduce = False
zoom_level     = 1.0
cam_offset_x    = 0
cam_offset_y    = 0
MAX_STEPS = 1000
MAX_IDLE = 60
predator_group = pygame.sprite.Group()
prey_group = pygame.sprite.Group()
predator_base_size = 12
predator_vision_distance = 120
predator_fertility = 0.9
dragging_vision_slider = False
dragging_size_slider = False
dragging_fertility_slider = False
tunnel_offset     = 0      
tunnel_speed      = 1.2
tunnel_spacing    = 40   
tunnel_line_width = 2      
tunnel_color      = (60,60,60)
end_screen_stats = {}

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

def load_save_data():
    # If missing, create a fresh file with three default slots
    if not os.path.exists(SAVE_FILE_PATH):
        data = {i: DEFAULT_SLOT.copy() for i in (1,2,3)}
        save_save_data(data)
        return data

    data = {}
    with open(SAVE_FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["slot"])
            data[idx] = {
                "used":       row["used"].lower() == "true",
                "XP":         int(row["XP"]),
                "size_min":   int(row["size_min"]),
                "size_max":   int(row["size_max"]),
                "vision_min": int(row["vision_min"]),
                "vision_max": int(row["vision_max"]),
                "fertility_min": float(row["fertility_min"]),
                "fertility_max": float(row["fertility_max"]),
            }

    # Ensure every slot 1–3 exists
    for i in (1,2,3):
        if i not in data:
            data[i] = DEFAULT_SLOT.copy()
    return data


def save_save_data(data):
    # Atomically rewrite the file with all three slots in order
    with open(SAVE_FILE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "slot","used","XP",
            "size_min","size_max",
            "vision_min","vision_max",
            "fertility_min","fertility_max"
        ])
        for slot in (1,2,3):
            sd = data.get(slot, DEFAULT_SLOT)
            writer.writerow([
                slot,
                sd["used"],
                sd["XP"],
                sd["size_min"],
                sd["size_max"],
                sd["vision_min"],
                sd["vision_max"],
                sd["fertility_min"],
                sd["fertility_max"]
            ])

# Global save state
save_data = load_save_data()

def create_tunnel_background():
    """
    Call once per frame when on the title screen.
    Fills the screen black, then draws concentric circle outlines
    whose radii march outward to simulate flying through a tunnel.
    """
    global tunnel_offset

    # 1) Black background
    screen.fill((0,0,0))

    # 2) Compute center & maximum radius we need to cover corners
    cx, cy = screen_width // 2, screen_height // 2
    max_radius = int(((screen_width**2 + screen_height**2)**0.5) / 2) + tunnel_spacing

    # 3) Advance the offset & wrap
    tunnel_offset = (tunnel_offset + tunnel_speed) % tunnel_spacing

    # 4) Draw rings every tunnel_spacing pixels
    r = tunnel_offset
    while r < max_radius:
        pygame.draw.circle(
            screen,
            tunnel_color,
            (cx, cy),
            int(r),
            tunnel_line_width
        )
        r += tunnel_spacing



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
        self.fertility = max(1, float(predator_fertility * random.uniform(0.9, 1.1)))
        self.children_count = 0
        self.disabled = False

        # vision & hunger
        self.vision_distance = max(1, int(predator_vision_distance * variation)) + self.size
        max_vision = 300.0
        vd = min(self.vision_distance, max_vision)
        max_fov = math.pi
        self.fov = max_fov * (1 - vd / max_vision)
        self.max_hunger      = self.size * random.randint(70, 90)
        self.hunger          = self.max_hunger
        self.energy_usage    = self.size * 0.05


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
        if self.vision_distance > 270:
            self.image = pygame.image.load("assets/cyclopspred.png").convert_alpha()
        elif self.disabled:
            self.image = pygame.image.load("assets/disabledpred.png").convert_alpha()
        else:
            self.image = pygame.image.load("assets/predimg.png").convert_alpha()
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
        f = self.fertility
        base = int(f)
        fractional = f - base 
        extra = 1 if random.random() < fractional else 0
        count = base + extra
        
        excess = max(0, count - 3)
        disabled_chance = min(1.0, 0.3 * excess)
        print(f"[DEBUG] Repro count={count}, disabled_chance={disabled_chance:.2f}")
        
        for _ in range(count):
            disabled = (random.random() < disabled_chance)
            if disabled:
                print(f"[DEBUG] Predator {self.name} disabled child")
            child = predator(self.x, self.y)
            for attr in ('size', 'base_speed', 'max_hunger', 'energy_usage', 'fov',
                        'vision_distance', 'max_chase_speed', 'chase_acceleration'):
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    factor = random.uniform(0.2, 0.6) if disabled else random.uniform(0.9, 1.1)
                    mutated = round(val * factor)
                    if attr == 'size':
                        mutated = val
                    setattr(child, attr, int(mutated) if attr in ('size', 'steps_remaining') else mutated)
            child.fertility = self.fertility * random.uniform(0.9, 1.1)
                
            if disabled:
                child.fertility = 0.0
                child.growth_rate = 0.2
                child.energy_usage = child.size * 0.5
                child.base_speed = 0.1
                child.max_hunger = child.max_hunger * random.uniform(0.1, 0.2)
                child.disabled  = True
                img = pygame.image.load("assets/disabledpred.png").convert_alpha()
                img = pygame.transform.smoothscale(img, (int(child.size*2), int(child.size*2)))
                child.original_image = img
                child.image          = img
            else:
                child.fertility = self.fertility * random.uniform(0.9, 1.1)
                child.disabled  = False
                    
            child.just_spawned = True
            child.hunger = child.max_hunger * (2/3)
            child.rect = child.image.get_rect(center=(child.x, child.y))
            children.append(child)
        self.action = IDLE
        self.children_count += len(children)
        return children
        


    def update(self):
        global controlling_mode, player_controlled, spectate_target, zoom_level
        self.age_seconds += time_multiplier * (1/60)
        self.update_growth_stats()
        if self.age_seconds >= MAX_AGE:
            if self == player_controlled:
                controlling_mode = False
                spectate_target = None
                zoom_level = 1.0
                player_controlled = None
            self.kill()
            return
        if self.hunger <= 0:
            if self == player_controlled:
                controlling_mode = False
                spectate_target = None
                zoom_level = 1.0
                player_controlled = None
            self.kill()
            return
        if controlling_mode and self is player_controlled:
            if not self.alive():
                controlling_mode = False
                spectate_target = None
                zoom_level = 1.0
                player_controlled = None
            speed = self.max_chase_speed if sprinting else self.base_speed
            if sprinting:
                self.hunger -= speed * self.energy_usage * 2.5 * time_multiplier
            else:
                self.hunger -= speed * self.energy_usage * 1.5 * time_multiplier
            self.x += move_x * speed * time_multiplier
            self.y += move_y * speed * time_multiplier
            if move_x != 0 or move_y != 0:
                self.angle = math.atan2(move_y, move_x)
            if trying_eat:
                if isinstance(self, predator):
                    for prey in prey_group:
                        dx = prey.x - self.x
                        dy = prey.y - self.y
                        dist = math.hypot(dx, dy)
                        if dist < PREDATOR_EAT_RANGE:
                            prey.kill()
                            self.hunger = self.max_hunger
                            break
            if trying_reproduce:
                if isinstance(self, predator):
                    if self.age_seconds >= PREDATOR_REPRO_AGE and self.hunger >= self.max_hunger / 2:
                        self.hunger -= self.max_hunger / 3
                        for child in self.reproduce():
                            predator_group.add(child)
                        
            self.x = max(10, min(screen_width-10, self.x))
            self.y = max(10, min(screen_height-10, self.y))
            self.rect.center = (int(self.x), int(self.y))
            return
            
        if hasattr(self, 'just_spawned') and self.just_spawned:
            self.action = WANDER
            self.moving = True
            self.steps_remaining = random.randint(30, 100)
            ang = random.uniform(0, 2 * math.pi)
            self.angle = ang
            self.direction_x = math.cos(ang)
            self.direction_y = math.sin(ang)
            del self.just_spawned
                        

        if self.chasing and self.target_prey:
            dx = self.target_prey.x - self.x
            dy = self.target_prey.y - self.y
            dist = math.hypot(dx, dy)
            if dist < 15:
                self.lost_counter = 0
                # eat
                self.target_prey.kill()
                self.hunger = self.max_hunger
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


    def draw_vision_mask(self, screen):
        # pull in your camera/zoom globals
        global cam_offset_x, cam_offset_y, zoom_level

        # 1) Compute the world‐to‐screen transform for the apex
        sx = int(self.x * zoom_level + cam_offset_x)
        sy = int(self.y * zoom_level + cam_offset_y)

        # 2) Scale vision_distance by zoom as well
        sd = self.vision_distance * zoom_level

        half = self.fov / 2
        # 3) Compute screen‐space cone edges
        left = (
            int(sx + math.cos(self.angle - half) * sd),
            int(sy + math.sin(self.angle - half) * sd)
        )
        right = (
            int(sx + math.cos(self.angle + half) * sd),
            int(sy + math.sin(self.angle + half) * sd)
        )

        # 4) Build the overlay and darken it
        mask = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 240))

        # 5) Punch out your triangle _in screen coords_
        pygame.draw.polygon(mask, (0,0,0,0), [(sx, sy), left, right])

        # 6) Blit on top
        screen.blit(mask, (0,0))


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
        self.image = pygame.image.load("assets/preyimg.png").convert_alpha()
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
        
def blit_world_with_camera(skip=(None)):
    """Draw the world to a temp surface, scale & blit it under camera."""
    global cam_offset_x, cam_offset_y
    # 1) draw into world_surf
    world_surf = background.copy()
    plant_group.draw(world_surf)
    for pr in prey_group:
        if pr is not skip:
            pr.custom_draw(world_surf, show_flightzone)
    for p in predator_group:
        if p is not skip:
            p.custom_draw(world_surf, prey_group, show_debug, show_vision)
        

    # 2) centre on spectate_target
    target = player_controlled if controlling_mode and player_controlled else spectate_target
    if target:
        cx, cy = target.x, target.y
        cam_offset_x = screen_width/2  - cx * zoom_level
        cam_offset_y = screen_height/2 - cy * zoom_level
    else:
        cam_offset_x = cam_offset_y = 0

    #clamp
    sw = int(screen_width * zoom_level)
    sh = int(screen_height * zoom_level)
    cam_offset_x = max(min(cam_offset_x, 0), screen_width  - sw)
    cam_offset_y = max(min(cam_offset_y, 0), screen_height - sh)
    cam_offset_x = int(cam_offset_x)
    cam_offset_y = int(cam_offset_y)
    
    scaled = pygame.transform.smoothscale(world_surf, (sw, sh))
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
    create_tunnel_background()
    title = title_font.render("Predator / Prey Simulation", True, (255,255,255))
    orig_img   = pygame.image.load("assets/start_button2.png").convert_alpha()
    orig_w, h  = orig_img.get_size()
    scale      = 2
    
    start_image = pygame.transform.smoothscale(
    orig_img,
    (orig_w * scale, h * scale)
    )
    
    start_button = start_image.get_rect(
    center=(
        screen_width//2,
        screen_height//2 + 200
    ))
    
    screen.blit(title, (screen_width//2 - title.get_width()//2, screen_height//2 - title.get_height()//2))
    screen.blit(start_image, start_button.topleft)
    pygame.draw.rect(screen, (0,0,0), start_button,2)
    
    # back button
    pygame.draw.rect(screen, (50,50,50), back_button)
    pygame.draw.rect(screen, (255,255,255), back_button, 2)
    txt = font.render("Back", True, (255,255,255))
    screen.blit(txt, (back_button.x + (back_button.width - txt.get_width())//2,
                    back_button.y + (back_button.height - txt.get_height())//2))


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
    create_tunnel_background()
    title = title_font.render(message, True, (255,255,255))
    screen.blit(title, (screen_width//2 - title.get_width()//2, 100 - title.get_height()//2))
    draw_population_graph(screen, x=275, y=150, w=550, h=250)
    start_button.x = screen_width//2 - 50
    start_button.y = screen_height - 100
    pygame.draw.rect(screen, (50,50,50), start_button)
    pygame.draw.rect(screen, (255,255,255), start_button, 2)
    txt = font.render("Restart", True, (255,255,255))
    screen.blit(txt, (start_button.x + (start_button.width - txt.get_width())//2,
                    start_button.y + (start_button.height - txt.get_height())//2))
    
    for key, value in end_screen_stats.items():
        txt = font.render(f"{key} {value}", True, (255,255,255))
        screen.blit(txt, (screen_width - 220, 200 + 20*list(end_screen_stats.keys()).index(key)))
    
def create_pregame_ui():
    # 1) background
    create_tunnel_background()

    # 2) centre X
    center_x = screen_width // 2

    # 3) Draw Start button centred
    #    (assumes start_image and start_button Rect exist)
    start_button.centerx = center_x
    screen.blit(start_image, start_button.topleft)
    pygame.draw.rect(screen, (50,50,50), start_button)
    pygame.draw.rect(screen, (255,255,255), start_button, 2)
    txt = font.render("Start", True, (255,255,255))
    screen.blit(
        txt,
        (start_button.x + (start_button.width - txt.get_width())//2,
        start_button.y + (start_button.height - txt.get_height())//2)
    )

    # 4) Common slider width & horizontal origin
    slider_w = 400
    slider_h = 8
    slider_x = screen_width//2 - slider_w//2
    slider_y = 300

    

    # --- Size slider ---
    size_slider_y = 300
    size_slider_h = 8
    size_slider_rect = pygame.Rect(
        slider_x, size_slider_y,
        slider_w, slider_h
    )
    screen.blit(track_img, size_slider_rect.topleft)
    sd = save_data[active_save_slot]
    size_min, size_max = sd["size_min"], sd["size_max"]
    pct = (predator_base_size - size_min) / (size_max - size_min)
    handle_x = slider_x + int(pct * slider_w)
    sh = size_handle_img.get_height()
    sw = size_handle_img.get_width()
    hx = handle_x - sw//2
    hy = size_slider_y + size_slider_h//2 - sh//2
    screen.blit(size_handle_img, (hx, hy))
    txt = font.render(f"Predator Size: {predator_base_size}", True, (255,255,255))
    screen.blit(
        txt,
        ((screen_width - txt.get_width())//2, size_slider_y - 30)
    )

    # --- Vision slider ---
    vision_slider_y = 220
    vision_slider_h = 8
    vision_slider_rect = pygame.Rect(
        slider_x, vision_slider_y,
        slider_w, slider_h
    )
    screen.blit(track_img, vision_slider_rect.topleft)
    vis_min, vis_max = sd["vision_min"], sd["vision_max"]
    pct = (predator_vision_distance - vis_min) / (vis_max - vis_min)
    handle_x = slider_x + int(pct * slider_w)
    vh = vision_handle_img.get_height()
    vw = vision_handle_img.get_width()
    hx = handle_x - vw//2
    hy = vision_slider_y + vision_slider_h//2 - vh//2
    screen.blit(vision_handle_img, (hx, hy))
    txt = font.render(f"Predator Vision: {predator_vision_distance}", True, (255,255,255))
    screen.blit(
        txt,
        ((screen_width - txt.get_width())//2, vision_slider_y - 30)
    )

    # --- Fertility slider ---
    fertility_slider_y = 380
    fertility_slider_h = 8
    fertility_slider_rect = pygame.Rect(
        slider_x, fertility_slider_y,
        slider_w, slider_h
    )
    screen.blit(track_img, fertility_slider_rect.topleft + (0, 80))
    fert_min, fert_max = sd["fertility_min"], sd["fertility_max"]
    pct = (predator_fertility - fert_min) / (fert_max - fert_min)
    handle_x = slider_x + int(pct * slider_w)
    fh = fertility_handle_img.get_height()
    fw = fertility_handle_img.get_width()
    hx = handle_x - fw//2
    hy = fertility_slider_y + fertility_slider_h//2 - fh//2
    screen.blit(fertility_handle_img, (hx, hy))
    txt = font.render(f"Predator Fertility: {predator_fertility}", True, (255,255,255))
    screen.blit(
        txt,
        ((screen_width - txt.get_width())//2, fertility_slider_y - 30)
    )

    # --- Locks (also centred over their slider) ---
    lock = pygame.image.load("assets/locked.png").convert_alpha()
    if sd['XP'] < VISION_UNLOCK_XP:
        # draw lock above size slider
        lx = center_x - lock.get_width() // 2
        ly = size_slider_y - 32
        screen.blit(lock, (lx, ly))
        
        txt = font.render(f"XP {sd['XP']}/{VISION_UNLOCK_XP}", True, (255,255,255))
        padding = 10
        txt_x = lx + lock.get_width() + padding
        lock_cy = ly + lock.get_height() // 2
        txt_y = lock_cy - txt.get_height() // 2
        screen.blit(txt, (txt_x, txt_y))
    if sd['XP'] < FERTILITY_UNLOCK_XP:
        # draw lock above fertility slider
        lx = center_x - lock.get_width() // 2
        ly = fertility_slider_y - 32
        screen.blit(lock, (lx, ly))
        
        txt = font.render(f"XP {sd['XP']}/{FERTILITY_UNLOCK_XP}", True, (255,255,255))
        padding = 10
        txt_x = lx + lock.get_width() + padding
        lock_cy = ly + lock.get_height() // 2
        txt_y = lock_cy - txt.get_height() // 2
        screen.blit(txt, (txt_x, txt_y))
    # back button
    pygame.draw.rect(screen, (50,50,50), back_button)
    pygame.draw.rect(screen, (255,255,255), back_button, 2)
    txt = font.render("Back", True, (255,255,255))
    screen.blit(txt, (back_button.x + (back_button.width - txt.get_width())//2,
                    back_button.y + (back_button.height - txt.get_height())//2))

def create_save_file_ui():
    create_tunnel_background()
    title = title_font.render("Load/Save Game", True, (255,255,255))
    screen.blit(title, (screen_width//2 - title.get_width()//2, 100))
    # Draw slot buttons
    for idx, btn in enumerate(save_file_buttons, start=1):
        pygame.draw.rect(screen, (50,50,50), btn)
        pygame.draw.rect(screen, (255,255,255), btn, 2)
        if save_data[idx]['used']:
            label = f"XP: {save_data[idx]['XP']}"
            txt = font.render(label, True, (255,255,255))
            screen.blit(txt, (
                btn.x + (btn.width - txt.get_width())//2,
                btn.y + (btn.height - txt.get_height())//2 + 10
            ))
            label = f"Save file {idx}" if save_data[idx]['used'] else "No save file"
            txt = font.render(label, True, (255,255,255))
            screen.blit(txt, (
                btn.x + (btn.width - txt.get_width())//2,
                btn.y + (btn.height - txt.get_height())//2 - 10
            ))
        else:
            label = f"Save file {idx}" if save_data[idx]['used'] else "No save file"
            txt = font.render(label, True, (255,255,255))
            screen.blit(txt, (
                btn.x + (btn.width - txt.get_width())//2,
                btn.y + (btn.height - txt.get_height())//2
            ))
        for idx, btn in enumerate(delete_save_button, start=1):
            if save_data[idx]['used']:
                pygame.draw.rect(screen, (50,50,50), btn)
                pygame.draw.rect(screen, (255,255,255), btn, 2)
                txt = font.render("X", True, (255,0,0))
                screen.blit(txt, (
                    btn.x + (btn.width - txt.get_width())//2,
                    btn.y + (btn.height - txt.get_height())//2
                ))
    # Back button



# // SPRITE GROUPS \\ #
predator_group = pygame.sprite.Group()
prey_group     = pygame.sprite.Group()
plant_group    = pygame.sprite.Group()

# // UI ELEMENTS \\ #
fertility_handle_img = pygame.image.load("assets/fertility_handle.png").convert_alpha()
fertility_handle_img = pygame.transform.smoothscale(
    fertility_handle_img,
    (24, 24) 
)
size_handle_img = pygame.image.load("assets/size_handle.png").convert_alpha()
size_handle_img = pygame.transform.smoothscale(
    size_handle_img,
    (24, 24) 
)
vision_handle_img = pygame.image.load("assets/vision_handle2.png").convert_alpha()
track_img = pygame.image.load("assets/slider.png").convert_alpha()
slow_down_button = pygame.Rect(screen_width-200, screen_height-50, 90, 40)
start_image = pygame.image.load("assets/start_button2.png").convert_alpha()
start_button = start_image.get_rect(center=(
    screen_width//2,
    screen_height//2 + 200
))
speed_up_button  = pygame.Rect(screen_width-100, screen_height-50, 90, 40)
settings_button  = pygame.Rect(screen_width-100, 10, 90, 40)
settings_button_title  = pygame.Rect(screen_width-100, 10, 90, 40)
back_button      = pygame.Rect(screen_width-100, 10, 90, 40)
restart_button = pygame.Rect(screen_width-100, screen_height-100, 90, 40)
delete_save_button = [
    pygame.Rect(screen_width//2 + 160, 220 + (i-1)*100, 30, 30)
    for i in range(1, 4)]
save_file_buttons = [
    pygame.Rect(screen_width//2 - 150, 200 + (i-1)*100, 300, 70)
    for i in range(1, 4)]
settings_checkboxes = [
    {"label":"Show Vision Cone", "rect":pygame.Rect(100,150,20,20), "checked":False},
    {"label":"Enable Chase Lines","rect":pygame.Rect(100,200,20,20), "checked":True},
    {"label":"Show Flight Zone","rect":pygame.Rect(100,250,20,20), "checked":True},
]


decision_interval   = 20
frame_counter       = 0

# // MAIN LOOP \\ #
def run_game():
    global time_multiplier, spectate_target, zoom_level, show_debug, show_vision 
    global show_flightzone, predator_base_size, predator_vision_distance, dragging_size_slider
    global gif_timer, gif_index, dragging_vision_slider, predator_counts, prey_counts 
    global time_steps, plant_spawn_timer, active_save_slot, message, predator_fertility
    global dragging_fertility_slider, end_screen_stats, player_controlled, controlling_mode
    global move_x, move_y, sprinting, trying_eat, trying_reproduce

    # now define your own counter for this run
    frame_counter = 0
    running       = True
    currscreen    = 'savefiles'
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if currscreen == 'gamescreen':
                    if event.key == pygame.K_ESCAPE:
                        spectate_target = None
                        zoom_level = 1.0
                        player_controlled = None
                        controlling_mode   = False
                    elif event.key == pygame.K_LEFT:
                        time_multiplier = max(0.5, time_multiplier/2)
                    elif event.key == pygame.K_RIGHT:
                        time_multiplier = min(16, time_multiplier*2)
                    elif event.key == pygame.K_c and spectate_target:
                        player_controlled = spectate_target
                        controlling_mode   = True
                    if event.key in (pygame.K_a, pygame.K_LEFT):  move_x = -1
                    if event.key in (pygame.K_d, pygame.K_RIGHT): move_x = +1
                    if event.key in (pygame.K_w, pygame.K_UP):    move_y = -1
                    if event.key in (pygame.K_s, pygame.K_DOWN):  move_y = +1
                    if event.key == pygame.K_LSHIFT:              sprinting = True
                    if event.key == pygame.K_e:                   trying_eat = True
                    if event.key == pygame.K_r:                   trying_reproduce = True
            elif event.type == pygame.KEYUP:
                if currscreen == 'gamescreen':
                    if event.key in (pygame.K_a, pygame.K_LEFT, pygame.K_d, pygame.K_RIGHT): move_x = 0
                    if event.key in (pygame.K_w, pygame.K_UP,   pygame.K_s, pygame.K_DOWN):  move_y = 0
                    if event.key == pygame.K_LSHIFT:             sprinting = False
                    if event.key == pygame.K_e:                  trying_eat = False
                    if event.key == pygame.K_r:                  trying_reproduce = False
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
                            message = "Simulation Terminated"
                            screen.fill((0,0,0))
                elif currscreen == 'title':
                    if start_button.collidepoint((mx, my)):
                        currscreen = 'pregame'
                    elif back_button.collidepoint((mx, my)):
                        currscreen = 'savefiles'
                        active_save_slot = None
                elif currscreen == "pregame":
                    size_slider_rect = pygame.Rect(300, 300, 400, 20)
                    vision_slider_rect = pygame.Rect(300, 220, 400, 20)
                    fertility_handle_rect = pygame.Rect(300, 380, 400, 20)
                    # slightly taller to make grabbing easier
                    if size_slider_rect.collidepoint(mx, my) and active_save_slot is not None and save_data[active_save_slot]['XP'] >= VISION_UNLOCK_XP:
                        dragging_size_slider = True
                    elif vision_slider_rect.collidepoint(mx, my):   
                        dragging_vision_slider = True
                    elif fertility_handle_rect.collidepoint(mx, my) and active_save_slot is not None and save_data[active_save_slot]['XP'] >= FERTILITY_UNLOCK_XP:
                        dragging_fertility_slider = True
                    elif start_button.collidepoint((mx, my)):
                        currscreen = 'gamescreen'
                        predator_counts = []
                        prey_counts = []
                        time_steps = []
                        simulation_seconds = 0
                        sd = save_data[active_save_slot]
                        session_size_min    = sd["size_min"]
                        session_size_max    = sd["size_max"]
                        session_vision_min  = sd["vision_min"]
                        session_vision_max  = sd["vision_max"]
                        session_fertility_min = sd["fertility_min"]
                        session_fertility_max = sd["fertility_max"]
                        spawn_initial_sprites()
                    elif back_button.collidepoint((mx, my)):
                        currscreen = 'title'
                elif currscreen == 'endscreen':
                    if start_button.collidepoint((mx, my)):
                        currscreen = 'pregame'
                        end_screen_stats = {}
                        prey_group.empty()
                        predator_group.empty()
                        plant_group.empty()
                        screen.fill((0,0,0))
                elif currscreen == 'savefiles':
                    for idx, btn in enumerate(save_file_buttons, start=1):
                        if btn.collidepoint((mx, my)):
                            if save_data[idx]['used'] == False:
                                save_data[idx]['used'] = True
                                save_save_data(save_data)
                                play_gif_sequence([
                                    "assets/story_scene1.gif",
                                    "assets/story_scene2.gif"
                                ])
                            active_save_slot = idx
                            predator_vision_distance = save_data[idx]["vision_min"]
                            predator_base_size = save_data[idx]["size_min"]
                            predator_fertility = save_data[idx]["fertility_min"]
                            currscreen = 'title'
                    for idx, btn in enumerate(delete_save_button, start=1):
                        if btn.collidepoint((mx, my)):
                            data = DEFAULT_SLOT
                            for key in data:
                                save_data[idx][key] = data[key]
                            save_save_data(save_data)
            elif event.type == pygame.MOUSEBUTTONUP:
                if currscreen == 'pregame':
                    dragging_vision_slider = False
                    dragging_size_slider = False
                    dragging_fertility_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
                if currscreen == 'pregame':
                    sd = save_data[active_save_slot]
                    if dragging_size_slider:
                        mx, my = pygame.mouse.get_pos()
                        rel_x = max(0, min(mx - 300, 400))
                        percent = rel_x / 400
                        predator_base_size = int(round(
                            sd["size_min"] + percent * (sd["size_max"] - sd["size_min"])
                        ))
                    elif dragging_vision_slider:
                        mx, my = pygame.mouse.get_pos()
                        rel_x = max(0, min(mx - 300, 400))
                        percent = rel_x / 400
                        predator_vision_distance = int(round(
                            sd["vision_min"] + percent * (sd["vision_max"] - sd["vision_min"])
                        ))
                    elif dragging_fertility_slider:
                        mx, my = pygame.mouse.get_pos()
                        rel_x = max(0, min(mx - 300, 400))
                        percent = rel_x / 400
                        predator_fertility = float(round(
                            sd["fertility_min"] + percent * (sd["fertility_max"] - sd["fertility_min"]),2
                        ))

        clock.tick(60)

        if currscreen=='gamescreen':
            screen.blit(background, (0,0))
            if len(predator_group) == 0 or len(prey_group) == 0:
                # choose your message
                if len(predator_group) == 0:
                    message = 'Predators have gone extinct!'
                    xp_gain = 1
                    end_screen_stats['Pity XP:'] = 1
                    end_screen_stats['Predators Left XP:'] = 0
                else:
                    message = 'Prey have gone extinct!'
                    xp_gain = len(predator_group) + 1
                    end_screen_stats['Pity XP:'] = 1
                    end_screen_stats['Predators Left XP:'] = len(predator_group)

                currscreen = 'endscreen'

                if active_save_slot is not None:
                    sd = save_data[active_save_slot]
                    # 1) bump XP
                    sd["XP"] += xp_gain

                    # 2) merge session bounds into slot
                    if session_vision_max > sd['vision_max']:
                        end_screen_stats['New Max Vision:'] = session_vision_max
                        
                    sd["vision_min"] = int(round(min(sd["vision_min"],  session_vision_min),0))
                    sd["vision_max"] = int(round(max(sd["vision_max"],  session_vision_max),0))
                    
                    if sd['XP'] >= VISION_UNLOCK_XP:
                        sd["size_min"]   = int(round(min(sd["size_min"],    session_size_min),0))
                        sd["size_max"]   = int(round(max(sd["size_max"],    session_size_max),0))
                        
                        if session_size_max > sd['size_max']:
                            end_screen_stats['New Max Size:'] = session_size_max
                            
                    if sd['XP'] >= FERTILITY_UNLOCK_XP:
                        sd["fertility_min"] = float(round(min(sd["fertility_min"], session_fertility_min),2))
                        sd["fertility_max"] = float(round(max(sd["fertility_max"], session_fertility_max),2))
                        
                        if session_fertility_max > sd['fertility_max']:
                            end_screen_stats['New Max Fertility:'] = session_fertility_max  

                    # 3) persist everything
                    save_save_data(save_data)

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
            for p in predator_group:
                session_size_min   = min(session_size_min, p.size)
                session_size_max   = max(session_size_max, p.size)
                session_vision_min = min(session_vision_min, p.vision_distance)
                session_vision_max = max(session_vision_max, p.vision_distance)
                session_fertility_max = max(session_fertility_max, p.fertility)
                session_fertility_min = min(session_fertility_min, p.fertility)
                
            prey_group.update()
            plant_group.update()

            # draws
            show_vision = next(opt for opt in settings_checkboxes
                            if opt["label"]=="Show Vision Cone")["checked"]
            show_debug  = next(opt for opt in settings_checkboxes
                            if opt["label"]=="Enable Chase Lines")["checked"]
            show_flightzone = next(opt for opt in settings_checkboxes
                            if opt["label"]=="Show Flight Zone")["checked"]
            blit_world_with_camera(skip=player_controlled)
            if controlling_mode and player_controlled:
                player_controlled.draw_vision_mask(screen)
                sx = int(player_controlled.x * zoom_level + cam_offset_x)
                sy = int(player_controlled.y * zoom_level + cam_offset_y)
                deg = -math.degrees(player_controlled.angle) - 90
                rotated = pygame.transform.rotate(player_controlled.original_image, deg)
                w, h = rotated.get_size()
                surf = pygame.transform.smoothscale(rotated, (int(w * zoom_level), int(h * zoom_level)))
                rect = surf.get_rect(center=(sx, sy))
                screen.blit(surf, rect)
                
                if show_debug:
                    for prey in prey_group:
                        px = int(prey.x * zoom_level + cam_offset_x)
                        py = int(prey.y * zoom_level + cam_offset_y)
                        pygame.draw.line(screen, (60,60,60), (sx, sy), (px, py), 1)
                
                if show_vision:
                    # screen‐space vision cone points
                    vd   = player_controlled.vision_distance * zoom_level
                    half = player_controlled.fov / 2
                    a    = player_controlled.angle

                    left = (
                        int(sx + math.cos(a - half) * vd),
                        int(sy + math.sin(a - half) * vd)
                    )
                    right = (
                        int(sx + math.cos(a + half) * vd),
                        int(sy + math.sin(a + half) * vd)
                    )

                    pygame.draw.polygon(
                        screen,
                        (200,200,200),
                        [(sx, sy), left, right],
                        1
                    )
                
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
        
        elif currscreen == 'savefiles':
            create_save_file_ui()     

        pygame.display.flip()
        frame_counter += 1

if __name__ == "__main__":
    run_game()
