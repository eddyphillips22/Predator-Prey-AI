# AI.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ——— match these dims to train_agents.py ———
PRED_STATE_DIM = 8
PREY_STATE_DIM = 8
PRED_ACTIONS   = 5
PREY_ACTIONS   = 5

class AgentNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,      64)
        self.out = nn.Linear(64, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# instantiate
pred_net = AgentNet(PRED_STATE_DIM, PRED_ACTIONS)
prey_net = AgentNet(PREY_STATE_DIM, PREY_ACTIONS)

# load your trained weights if they exist
if os.path.isfile("predator_net.pth"):
    ckpt = torch.load("predator_net.pth")
    model_dict = pred_net.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered)
    pred_net.load_state_dict(model_dict)
if os.path.isfile("prey_net.pth"):
    ckpt = torch.load("prey_net.pth")
    model_dict = prey_net.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered)
    prey_net.load_state_dict(model_dict)

pred_net.eval()
prey_net.eval()

def pred_q_action(state_vec, self):
    """
    state_vec: 6-element list or np.array
    returns: integer action ∈ [0,1,2,3]
    """

    t = torch.tensor(state_vec, dtype=torch.float32)
    with torch.no_grad():
        qs = pred_net(t)
    q = qs.squeeze(0).cpu().numpy()
    
    from game_loop import PREDATOR_REPRO_AGE, MAX_AGE, REPRODUCE, predator as pred
    age_norm = state_vec[6]
    threshold_norm = PREDATOR_REPRO_AGE / MAX_AGE
    hunger_norm = state_vec[1]
    hunger_threshold_norm = self.max_hunger / 2 / self.max_hunger
    if age_norm < threshold_norm or hunger_norm < hunger_threshold_norm:
        q[REPRODUCE] = -np.inf
        
    return int(q.argmax())

def prey_q_action(state_vec, self):
    t = torch.tensor(state_vec, dtype=torch.float32)
    with torch.no_grad():
        qs = prey_net(t)
    q = qs.squeeze(0).cpu().numpy()
    
    from game_loop import PREY_REPRO_AGE, MAX_AGE, P_REPRODUCE, prey
    age_norm = state_vec[6]
    age_threshold_norm = PREY_REPRO_AGE / MAX_AGE
    hunger_norm = state_vec[1]
    hunger_threshold_norm = self.max_hunger / 2 / self.max_hunger

    if age_norm < age_threshold_norm or hunger_norm < hunger_threshold_norm:
        q[P_REPRODUCE] = -np.inf
        
    return int(q.argmax())
