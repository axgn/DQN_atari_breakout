import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

gym.register_envs(ale_py)

env_name = "ALE/Breakout-v5"
env = gym.make(env_name)
print("action:", env.unwrapped.get_action_meanings())
