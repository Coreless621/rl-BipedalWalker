import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models import Actor, Critic
from replaybuffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise, Gaussian_noise
from tqdm import tqdm

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        with torch.no_grad():
            target_param.copy_(tau * source_param + (1 - tau) * target_param)

def train(actor, critic, target_actor, target_critic, replay_buffer, actor_optim, critic_optim, gamma, tau, batch_size, max_norm):
    if replay_buffer.size < batch_size:
        return

    states, actions ,rewards, next_states, terminates = replay_buffer.sample(batch_size)
    
    # computing target q values
    with torch.no_grad():
        next_actions = target_actor(next_states)
        target_q_values = target_critic(next_states, next_actions)
    y = rewards + gamma * (1-terminates) * target_q_values

    # computing critic loss
    critic_q_value = critic(states, actions)
    critic_loss = torch.nn.functional.mse_loss(critic_q_value, y)

    # updating critic 
    critic_optim.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm)
    critic_optim.step()

    # computing actor loss
    actions = actor(states)
    actor_loss = -torch.mean(critic(states, actions))

    # updating actor 
    actor_optim.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm)
    actor_optim.step()

    # updating target networks
    soft_update(target_actor, actor, tau)
    soft_update(target_critic, critic, tau)

    return critic_loss.item(), actor_loss.item() # returned for tensorboard

def main():
    device = "cpu"
    num_envs = 3
    envs = gym.make_vec("BipedalWalker-v3", num_envs=num_envs, vectorization_mode="async")
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    gamma = 0.99
    tau = 0.004
    batch_size = 128
    storage_size = 1_000_000
    replay_buffer = ReplayBuffer(storage_size, obs_dim, action_dim, device)
    num_episodes = 10_000
    lr_actor = 1e-5
    lr_critic = 1e-4
    moving_avg_window = 50 # used for logging average rewards (last 50 episodes) 
    min_sigma = 0.05
    initial_sigma = 0.3
    sigma_decay = (min_sigma/initial_sigma)**(1/6250) # decays sigma until 6250 episodes
    ou_noise = OrnsteinUhlenbeckNoise(size=(action_dim,), mu=0.0, theta=0.15, sigma=initial_sigma, dt=1e-2)
    gaussian_noise = Gaussian_noise((action_dim), std=initial_sigma)
    max_norm = 1.0 # a value used for gradient clipping


    # initializing networks
    actor = Actor(obs_dim, action_dim).to(device)
    target_actor = Actor(obs_dim, action_dim).to(device)
    critic = Critic(obs_dim, action_dim).to(device)
    target_critic = Critic(obs_dim, action_dim).to(device)

    try:
        actor.load_state_dict(torch.load("actor.pth"))
        target_actor.load_state_dict(torch.load("target_actor.pth"))
        critic.load_state_dict(torch.load("critic.pth"))
        target_critic.load_state_dict(torch.load("target_critic.pth"))
    except FileNotFoundError:
        tqdm.write("No Models found. Creating new Models.")


    # setting optimizers
    actor_optim = optim.Adam(actor.parameters(), lr_actor)
    critic_optim = optim.Adam(critic.parameters(), lr_critic)

    training_rewards = [] # collecting episode rewards over training - used to compute avg reward for tensorboard
    critic_losses = []
    actor_losses = []
    
    writer = SummaryWriter("runs/BipedalWalker_exp12")
    
    pbar = tqdm(total=num_episodes, desc="Training progress") # progression bar
    episode_count = 0
    global_step = 0
    done = False
    states, _ = envs.reset()
    episode_reward = np.zeros(num_envs) # logs accumulated reward for each env per episode

    while not done:
        #ou_noise.reset()

        state_tensors = torch.tensor(states, dtype=torch.float32).to(device)
        actions = actor(state_tensors).detach().cpu().numpy()
        noise = gaussian_noise()
        noisy_actions = actions + noise
        noisy_actions = np.clip(noisy_actions, -1, 1)
        
        next_states, rewards, terminates, truncates, _ = envs.step(noisy_actions)
        global_step+=1
        episode_reward += rewards

        # filling buffer, checking for episode end and adding respective reward
        for i in range(num_envs): 
            replay_buffer.push(states[i], noisy_actions[i], rewards[i], next_states[i], terminates[i])
            if terminates[i] or truncates[i]:
                episode_count+=1
                training_rewards.append(episode_reward[i])
                episode_reward[i]=0.0
                pbar.update(episode_count - pbar.n)
    	
        gaussian_noise.std = max(min_sigma, gaussian_noise.std * sigma_decay)
        
        states = next_states
        if episode_count >= num_episodes: # handling training end 
            done = True

        losses = train(actor, critic, target_actor, target_critic, replay_buffer, actor_optim, critic_optim, gamma, tau, batch_size, max_norm)
        
        # logging part 
        if losses is not None:
            critic_loss, actor_loss = losses
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

        try:
            avg_reward = np.mean(training_rewards[-moving_avg_window:]) 
            writer.add_scalar("Average Reward", avg_reward, global_step)
        except RuntimeWarning: # if mean of empty slice
            avg_reward = np.mean(training_rewards)
            writer.add_scalar("Average Reward", avg_reward)

        if len(critic_losses) > 0: # checking because losses only are generated if the buffer has batch_size samples; see train()
            try:
                avg_actor_loss = -np.mean(actor_losses[-moving_avg_window:])
                avg_critic_loss = np.mean(critic_losses[-moving_avg_window:])
                writer.add_scalar("Average Actor Loss", avg_actor_loss, global_step) 
                writer.add_scalar("Average Critic Loss", avg_critic_loss, global_step)
            except RuntimeWarning:
                avg_actor_loss = -np.mean(actor_losses)
                avg_critic_loss = np.mean(critic_losses)
                writer.add_scalar("Average Actor Loss", avg_actor_loss, global_step)
                writer.add_scalar("Average Critic Loss", avg_critic_loss, global_step) 


        if (global_step) % 1000 == 0: # logging histograms every 10 episodes
            for name, params in actor.named_parameters():
                if params.grad is not None:
                    writer.add_histogram(f"actor{name}/grad", params.grad, global_step)

            for name, params in critic.named_parameters():
                if params.grad is not None:
                    writer.add_histogram(f"critic{name}/grad", params.grad, global_step)
        
    
    torch.save(actor.state_dict(), "actor.pth")
    torch.save(critic.state_dict(), "critic.pth")
    torch.save(target_actor.state_dict(), "target_actor.pth")
    torch.save(target_critic.state_dict(), "target_critic.pth")
    tqdm.write("Model saved.")
    
    writer.close()
    pbar.close()
    print("Training completed.")
    

if __name__ == "__main__":
    main()

