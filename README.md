# 🦿 BipedalWalker-v3 – DDPG Agent

https://github.com/user-attachments/assets/d086b078-fc48-40fc-a488-09c821ddd00f

This project implements a **Deep Deterministic Policy Gradient (DDPG)** agent to solve the `BipedalWalker-v3` environment from [Gymnasium](https://gymnasium.farama.org/).  
It uses continuous control, actor-critic architecture, noise exploration, and experience replay for learning.

---

## 🌍 Environment Overview

- **Environment:** `BipedalWalker-v3`



- **State space:** 24 continuous observations (robot position, joint angles, velocities, etc.)
- **Action space:** 4 continuous values (control torques)
- **Goal:** Train a bipedal robot to walk across terrain without falling
- **Reward:** Based on distance covered and stability

---

## 🧠 Algorithm: DDPG

- **Type:** Off-policy, model-free actor-critic algorithm for continuous control
- **Techniques used:**
  - Twin neural networks: `Actor` and `Critic`
  - Target networks and soft updates (Polyak averaging)
  - Replay buffer for experience reuse
  - Gradient clipping
  - Ornstein-Uhlenbeck or Gaussian noise for exploration

---

## ⚙️ Hyperparameters

| Parameter          | Value        |
|--------------------|--------------|
| Episodes           | `10,000`     |
| Actor LR           | `1e-5`       |
| Critic LR          | `1e-4`       |
| Gamma              | `0.99`       |
| Tau (soft update)  | `0.004`      |
| Batch size         | `128`        |
| Buffer size        | `1,000,000`  |
| Noise std (init)   | `0.3`        |
| Noise std (min)    | `0.05`       |

---

## 🧩 Components

| File             | Description |
|------------------|-------------|
| `models.py`      | Defines `Actor` and `Critic` neural networks |
| `replaybuffer.py`| Vectorized PyTorch replay buffer for efficient sampling |
| `noise.py`    | Implements Ornstein-Uhlenbeck and Gaussian noise generators |
| `training.py`    | Main DDPG training loop using vectorized environments and TensorBoard logging |
| `testing.py`     | Loads a trained actor and evaluates performance in `human` mode |
| `actor.pth`, `critic.pth`, etc. | Saved PyTorch weights for trained networks |

---

## 📊 Logging & Visualization

- Uses `TensorBoard` (`SummaryWriter`) to log:
  - Average rewards
  - Actor and critic loss
  - Parameter gradients

Launch logs with:

```bash
tensorboard --logdir=runs
