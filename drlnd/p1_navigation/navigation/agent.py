import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from navigation import device
from navigation.model import QNetwork
from navigation.memory import Memory

class Agent():
    '''Interacts with and learns from the environment.'''

    def __init__(self, state_size, action_size,
                    buffer_size=int(1e5), batch_size=64, update_freq=4,
                    gamma=0.99, tau=1e-3, lr=5e-4,
                    seed=0, checkpoint=None):
        '''
        Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            checkpoint (str): path to file containing model's state_dict
            buffer_size (int): number of records to stor in experience replay buffer
            batch_size (int): minibatch size for each epoch of neural network training
            update_freq (int): number of steps to run before updating target network
            gamma (float): RL discount factor
            tau (float): Q-Learning interpolation factor
            lr (float): Neural network learning rate
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        if checkpoint is not None:
            self.load(checkpoint)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=lr, momentum=0.9)
        # self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=lr)
        self.lossfn = F.mse_loss

        # Discount/interplation factors
        self.gamma = gamma
        self.tau = tau

        # Replay memory
        self.memory = Memory(action_size, buffer_size, batch_size, seed)
        
        # Initialize time step
        self.t_step = 0
        self.update_freq = update_freq
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_freq time steps.
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        '''
        Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        '''
        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        targets = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + (gamma * targets)
        expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = self.lossfn(expected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        '''
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def load(self, checkpoint):
        '''
        Load model weights from checkpoint file

        Params
        ======
            checkpoint (str): path to file containing model's state_dict
        '''
        print(f'Loading model weights from {checkpoint}')
        self.qnetwork_local.load_state_dict(torch.load(checkpoint, map_location=device))
        self.qnetwork_target.load_state_dict(torch.load(checkpoint, map_location=device))