# env_wrapper.py

import random
import math

# import your simulation classes *without* the Pygame rendering bits
from game_loop import predator, prey, plant, screen_width, screen_height, IDLE, P_IDLE, REPRODUCE, P_REPRODUCE, PREDATOR_REPRO_AGE, PREY_REPRO_AGE, MAX_AGE
# 1 episode max length
MAX_STEPS = 1000
MAX_IDLE = 60

class Env:
    def __init__(self):
        self.pred = None
        self.prey = None
        self.plants = []
        self.step_count = 0
        self._last_dist = None

    def reset(self):
        """Spawn fresh predator, prey, plants, zero timers. Return initial states."""
        # 1. create one predator and one prey
        self.pred = predator(random.randint(10, screen_width-10),
                            random.randint(10, screen_height-10))
        self.pred.age_seconds = PREDATOR_REPRO_AGE
        self.prey = prey(   random.randint(10, screen_width-10),
                            random.randint(10, screen_height-10))
        self.prey.age_seconds = PREY_REPRO_AGE
        # 2. spawn a few plants (optional for RL reward shaping)
        self.plants = [plant(random.randint(10, screen_width-10),
                            random.randint(10, screen_height-10))
                        for _ in range(10)]
        self.step_count = 0

        dx = self.prey.x - self.pred.x
        dy = self.prey.y - self.pred.y
        self._last_dist = math.hypot(dx, dy)
        # 3. compute and return the two initial state‐vectors
        return self._get_pred_state(), self._get_prey_state()

    def step(self, pred_action, prey_action):
        """
        pred_action/prey_action: integers ∈ {0..3}
        1) apply actions (override their .chasing/._flee_target flags)
        2) call .update() on pred, prey, plants
        3) compute rewards & done
        4) return (next_pred_state, next_prey_state, r_pred, r_prey, done)
        """
        # —— 1) apply predator action —— #
        if pred_action == 2:  # CHASE
            # force a chase: pick the prey as target
            self.pred.chasing     = True
            self.pred.target_prey = self.prey
        elif pred_action == 1:  # WANDER
            self.pred.chasing     = False
            self.pred.target_prey = None
        elif pred_action == 3:  # EAT
            # do nothing here—eat happens in update() when close enough
            pass
        elif pred_action == 4:
            if (self.pred.age_seconds >= PREDATOR_REPRO_AGE 
                and self.pred.hunger >= self.pred.max_hunger/2):
                self.pred.chasing     = False
                self.pred.target_prey = None
                self.pred.action = REPRODUCE
            else:
                self.pred.chasing     = False
                self.pred.target_prey = None
                self.pred.action = IDLE            
            
        else:  # IDLE
            self.pred.chasing     = False
            self.pred.target_prey = None
            
        visible = len(self.pred.detect_prey([self.prey])) > 0

        # —— 1b) apply prey action —— #
        if prey_action == 2:  # FLEE
            # force flee: set the predator as _flee_target
            self.prey._flee_target = self.pred
            self.prey.flee_score   = 1.0
            self.prey.action = 2
        elif prey_action == 3:  # GRAZE
            self.prey._flee_target = None
            self.prey.action = 3
        elif prey_action == 4:
            if (self.prey.age_seconds >= PREY_REPRO_AGE and
                self.prey.hunger >= self.prey.max_hunger/2):
                self.prey._flee_target = None
                self.prey.action = P_REPRODUCE
            else:
                self.prey._flee_target = None
                self.prey.action = IDLE
        
            

        # IDLE/WANDER: leave their normal idle‐wander logic intact

        # —— 2) step the simulation —— #
        pred_prev_children = self.pred.children_count
        prey_prev_children = self.prey.children_count
        old_dist = math.hypot(self.prey.x - self.pred.x,self.prey.y - self.pred.y)
        old_border = min(
            self.prey.x,
            screen_width  - self.prey.x,
            self.prey.y,
            screen_height - self.prey.y
        )

        self.step_count += 1

        # —— 3) compute rewards —— #
        # predator: +10 for eating, -1 per step, -10 if died
        r_pred = +0.4
        # detect if prey was killed this step:
        if not self.prey.alive():
            r_pred += 85
        if not self.pred.alive():
            r_pred -= 100
        if self.pred.children_count > pred_prev_children:
            r_pred += 10 * (self.pred.children_count - pred_prev_children)
        if self.pred.age_seconds >= PREDATOR_REPRO_AGE and self.pred.action != 4 and self.pred.hunger >= self.pred.max_hunger/2:
            r_pred -= 0.001 * self.pred.hunger
        
        if pred_action == 2 and visible: r_pred += 1.0
        elif pred_action == 2 and not visible: r_pred -= 1
        
        dx2 = self.prey.x - self.pred.x
        dy2 = self.prey.y - self.pred.y
        new_dist = math.hypot(dx2, dy2)
        delta = self._last_dist - new_dist
        if pred_action == 2:
            r_pred += 0.3 * delta
        self._last_dist = new_dist
        
        r_pred -= 0.05 * ((self.pred.max_hunger - self.pred.hunger) / self.pred.max_hunger)

        # prey: +1 per step survived, -50 if eaten, -0.1 per hunger unit
        r_prey = +0.2
        if not self.prey.alive():
            r_prey -= 100
        # hunger penalty:
        r_prey -= 0.2 * ((self.prey.max_hunger - self.prey.hunger) / self.prey.max_hunger)
        
        dx2 = self.prey.x - self.pred.x
        dy2 = self.prey.y - self.pred.y
        new_dist = math.hypot(dx2, dy2)
        delta = new_dist - old_dist
        
        if self.pred.chasing:
            if prey_action == 2: 
                r_prey += 10.0
                r_prey += 0.5 * delta
            elif prey_action in (0,1,3): 
                r_prey -= 10.0
        else:
            if prey_action == 3:
                hunger_norm = (self.prey.max_hunger - self.prey.hunger) / self.prey.max_hunger
                r_prey += 20.0 + 15.0 * hunger_norm
            elif prey_action in (0,1): r_prey -= 5.0
            elif prey_action == 2: r_prey -= 5.0
            
        if self.prey.children_count > prey_prev_children:
            r_prey += 100 * (self.prey.children_count - prey_prev_children)
        if self.prey.age_seconds >= PREY_REPRO_AGE and self.prey.action != 4 and self.prey.hunger >= self.pred.max_hunger/2:
            r_prey -= 0.002 * self.prey.hunger
        
        # plants grow
        
        
        new_border = min(
            self.prey.x,
            screen_width  - self.prey.x,
            self.prey.y,
            screen_height - self.prey.y
        )
        delta_border = new_border - old_border
        
        if prey_action == 2:   # FLEE
            # a) reward moving away from the wall
            border_coeff = 1.0    # crank this up—try 0.5–2.0
            r_prey += border_coeff * delta_border
            
        wall_thresh = 50      # pixels
        if new_border < wall_thresh:
            # linear penalty scaled by how deep into the “danger zone” they are
            pen_coeff = 5.0   # big stick
            r_prey -= pen_coeff * (wall_thresh - new_border) / wall_thresh
        
        if prey_action == 4 and self.prey.hunger >= self.prey.max_hunger/4 and self.prey.age_seconds >= PREY_REPRO_AGE:
            r_prey += 20
        elif prey_action == 4 and self.prey.hunger <= self.prey.max_hunger/4 and self.prey.age_seconds >= PREY_REPRO_AGE:
            r_prey -= 20
        
        self.pred.update()
        self.prey.update()
        for pl in self.plants:
            pl.update()
            
                

            
        

        # —— 4) done? —— #
        done = (self.step_count >= MAX_STEPS) or (not self.pred.alive()) or (not self.prey.alive())

        # —— 5) next states —— #
        next_pred_state = self._get_pred_state()
        next_prey_state = self._get_prey_state()

        return next_pred_state, next_prey_state, r_pred, r_prey, done

    # —— Helpers to build the 6-dim feature vectors —— #
    def _get_pred_state(self):
        # find prey dist
        dx, dy = self.prey.x - self.pred.x, self.prey.y - self.pred.y
        dist = math.hypot(dx, dy)
        age_norm = min(self.pred.age_seconds, MAX_AGE) / MAX_AGE
        return [
            min(dist, self.pred.vision_distance) / self.pred.vision_distance,
            self.pred.hunger / self.pred.max_hunger,
            self.pred.current_speed / self.pred.max_chase_speed,
            len(self.pred.detect_prey([self.prey])),
            self.pred.energy_usage,
            min(self.pred.time_since_eat, MAX_STEPS) / MAX_STEPS,
            age_norm,
            self.pred.children_count
        ]

    def _get_prey_state(self):
        # find preds in flight zone
        dx, dy = self.pred.x - self.prey.x, self.pred.y - self.prey.y
        dist = math.hypot(dx, dy)
        in_zone = dist <= self.prey.flightzone_radius
        age_norm = min(self.prey.age_seconds, MAX_AGE) / MAX_AGE
        plants_ds = []
        for pl in self.plants:
            dxp = pl.rect.centerx - self.prey.x
            dyp = pl.rect.centery - self.prey.y
            dp  = math.hypot(dxp, dyp)
            if dp <= self.prey.flightzone_radius:
                plants_ds.append(dp)
        if plants_ds:
            plant_norm = min(plants_ds) / self.prey.flightzone_radius
        else:
            plant_norm = 1.0
        
        return [
            min(dist, self.prey.flightzone_radius) / self.prey.flightzone_radius,
            1 - (self.prey.hunger / self.prey.max_hunger),
            self.prey.current_speed / self.prey.max_chase_speed,
            1 if in_zone and self.pred.chasing else 0,
            plant_norm,
            min(self.prey.time_since_idle, MAX_IDLE) / MAX_IDLE,
            age_norm,
            self.prey.children_count
        ]
