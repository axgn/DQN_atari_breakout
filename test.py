import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ale_py

gym.register_envs(ale_py)


env_name = "ALE/Breakout-v5"
env = gym.make(env_name)

print("action:", env.unwrapped.get_action_meanings())
print("observation space:", env.observation_space)
print("action space:", env.action_space)
print("metadata:", env.metadata)

state,_ = env.reset()
plt.imshow(state)
plt.savefig("figure_init.png")
action = env.action_space.sample()
print("action sample:", action)
state_next, reward,ter,tru,info = env.step(1)
plt.figure()
plt.imshow(state_next)
plt.savefig("figure.png")
