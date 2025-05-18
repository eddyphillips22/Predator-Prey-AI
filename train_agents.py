# train_agents.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import math
import game_loop
game_loop.time_multiplier = 8  

# ————— Hyperparameters —————
PRED_STATE_DIM = 8
PREY_STATE_DIM = 8
PRED_ACTIONS   = 5  # Idle, Wander, Chase, Eat
PREY_ACTIONS   = 5  # Idle, Wander, Flee, Graze

GAMMA      = 0.99
LR         = 1e-3
EPS_START  = 1.0
EPS_END    = 0.05
EPS_DECAY  = 0.9995
BATCH_SIZE = 64
MEM_CAP    = 10000
EPISODES   = 10000

# ————— Replay Buffer —————
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *transition):
        self.buf.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)
    def __len__(self):
        return len(self.buf)

# ————— Agent Networks —————
class AgentNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# ————— Dummy Env Wrapper —————
# You need to write `Env.step(pred_action, prey_action)` which:
#  - applies those actions to your predator/prey,
#  - steps one frame of your Game Loop’s update(),
#  - returns (pred_state_next, prey_state_next, r_pred, r_prey, done_flag).
#
# And Env.reset() should re-spawn everything. For brevity, that wrapper
# is omitted here—but you’ll link into your existing Game Loop code.

from env_wrapper import Env
def dqn_update(net, opt, batch, gamma):
    # 1) unpack
    states, actions, rewards, next_states, dones = zip(*batch)
    states      = torch.tensor(states, dtype=torch.float32)
    actions     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # 2) current Q-values Q(s,a)
    q_values = net(states).gather(1, actions)

    # 3) target Q: r + γ * max_a' Q(s',a') * (1 - done)
    with torch.no_grad():
        q_next = net(next_states).max(1)[0].unsqueeze(1)
        q_target = rewards + gamma * q_next * (1 - dones)

    # 4) loss and backward
    loss = F.mse_loss(q_values, q_target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

# ————— Training Loop —————
pred_buffer = ReplayBuffer(MEM_CAP)
prey_buffer = ReplayBuffer(MEM_CAP)
def train():
    env = Env()
    pred_net = AgentNet(PRED_STATE_DIM, PRED_ACTIONS)
    prey_net = AgentNet(PREY_STATE_DIM, PREY_ACTIONS)
    pred_opt = torch.optim.Adam(pred_net.parameters(), lr=LR)
    prey_opt = torch.optim.Adam(prey_net.parameters(), lr=LR)
    eps = EPS_START

    for ep in range(EPISODES):
        s_pred, s_prey = env.reset()
        done = False
        while not done:
            # ε-greedy actions
            if random.random() < eps:
                a_pred = random.randrange(PRED_ACTIONS)
                a_prey = random.randrange(PREY_ACTIONS)
            else:
                with torch.no_grad():
                    a_pred = pred_net(torch.tensor(s_pred, dtype=torch.float32)).argmax().item()
                    a_prey = prey_net(torch.tensor(s_prey, dtype=torch.float32)).argmax().item()

            s_pred2, s_prey2, r_pred, r_prey, done = env.step(a_pred, a_prey)
            dx = env.prey.x - env.pred.x
            dy = env.prey.y - env.pred.y
            dist = math.hypot(dx, dy)
            in_zone = dist <= env.prey.flightzone_radius
            pred_buffer.push(s_pred, a_pred, r_pred, s_pred2, done)
            prey_buffer.push(s_prey, a_prey, r_prey, s_prey2, done)
            s_pred, s_prey = s_pred2, s_prey2

            # learning step
            if len(pred_buffer) >= BATCH_SIZE:
                batch_pred = pred_buffer.sample(BATCH_SIZE)
                dqn_update(pred_net, pred_opt, batch_pred, GAMMA)
            if len(prey_buffer) >= BATCH_SIZE:
                batch_prey = prey_buffer.sample(BATCH_SIZE)
                dqn_update(prey_net, prey_opt, batch_prey, GAMMA)

        eps = max(EPS_END, eps * EPS_DECAY)

    # save final weights
    torch.save(pred_net.state_dict(), "predator_net.pth")
    torch.save(prey_net.state_dict(),   "prey_net.pth")
    print("Training complete.")

if __name__=="__main__":
    train()
