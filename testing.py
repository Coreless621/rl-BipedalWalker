import gymnasium as gym
from models import Actor
import numpy as np
import torch


actor = Actor(24, 4)
actor.load_state_dict(torch.load("actor.pth"))
env = gym.make("BipedalWalker-v3", render_mode = "human")

for _ in range(10):
    state, _ = env.reset()
    done = False
    timestep = 0
    episode_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_tensor).detach().numpy()[0]
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        timestep += 1
        done = terminated or truncated
        state = next_state
    print(f"Completed episode with {episode_reward} points in {timestep} timesteps.")

env.close()


