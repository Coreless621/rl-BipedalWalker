import torch

class ReplayBuffer():
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        
        self.states = torch.zeros(self.capacity, state_dim, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.capacity, action_dim, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.capacity, 1, dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros(self.capacity, state_dim, dtype=torch.float32, device=self.device)
        self.terminates = torch.zeros(self.capacity, 1, dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, termination):
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.terminates[self.ptr] = torch.as_tensor(termination, dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_terminates = self.terminates[indices]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminates



