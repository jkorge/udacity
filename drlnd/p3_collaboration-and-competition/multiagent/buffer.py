import pdb
from collections import deque
from dataclasses import dataclass

from multiagent import torch, device, RNG

@dataclass
class Trajectory:
    '''
    Collection of:
        - starting state
        - first action
        - total discounted reward
        - last state
        - terminal status
    of an N-step trajectory
    '''
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    last_state: torch.tensor
    done: torch.tensor

class PrioritizedReplayBuffer:
    '''Prioritized buffer for storing RL agent experience'''

    def __init__(self, state_size, action_size, trajectory_length, n_agents, gamma, buffer_size=1e6, batch_size=64):
        '''
        Initialize a PrioritizedReplayBuffer object.

        Parameters
        ----------
            state_size : int
                Dimensionality of agent's state
            action_size : int
                Dimensionality of action space
            trajectory_length : int
                Number of steps to accumulate in each trajectory
            n_agents : int
                Number of agents adding to the buffer
            gamma : float
                Discount factor
            buffer_size : int
                Maximum number of records to hold in buffer
            batch_size : int
                Size of batches returned by Memory.batch
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.trajectory_length = trajectory_length
        self.n_agents = n_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Containers for stored trajectories and associated priorities
        self.state_buffer = torch.zeros((self.buffer_size, self.state_size), device=device)
        self.action_buffer = torch.zeros((self.buffer_size, self.action_size), device=device)
        self.reward_buffer = torch.zeros(self.buffer_size, device=device)
        self.last_state_buffer = torch.zeros((self.buffer_size, self.state_size), device=device)
        self.done_buffer = torch.zeros(self.buffer_size, dtype=torch.bool, device=device)
        self.priorities = torch.zeros(self.buffer_size, device=device)

        # Next record to add data to
        self.pos = 0

        # Flag for when buffers are filled with records
        self.full = False

        # misc params
        self.max_priority = torch.tensor(1.0, device=device)
        self.beta = 1e-3
        self.eps = 1. + 1e-7

        self.reset()
    
    def add(self, states, actions, rewards, next_states, dones):
        '''
        Add new step to current trajectory. Adds trajectory to buffer when trajectory reaches length provided during construction

        Parameters
        ----------
            states : List[np.ndarray]
                Oberservations for each agent
            actions : List[np.ndarray]
                Actions each agent took from corresponding `states`
            rewards : np.ndarray or array_like
                Rewards each agent received for taking corresponding `actions`
            next_states : List[np.ndarray]
                Observations each agent made after taking corresponding `actions`
        '''

        if self.n_agents == 1:
            dones = dones.unsqueeze(0)
            rewards = rewards.unsqueeze(0)

        for i in range(self.n_agents):
            if not torch.any(dones[i]):
                # accumulate discounted rewards
                self.current_trajectory[i].reward += self.gamma**self.current_step * rewards[i]

            if self.current_step == 0:
                # save initial state/action
                self.current_trajectory[i].state= states[i]
                self.current_trajectory[i].action = actions[i]

            elif self.current_step == (self.trajectory_length - 1):
                # save N-th state and terminal status
                self.current_trajectory[i].last_state = next_states[i]
                self.current_trajectory[i].done = dones[i]

        self.current_step = (self.current_step + 1) % self.trajectory_length

        if self.current_step == 0:
            # end of trajectory
            self._append()
            self.reset()
            return True
        else:
            return False

    def reset(self):
        '''Replace current trajectory with zero-initialized arrays. Reset step counter'''

        self.current_step = 0
        self.current_trajectory = [
            Trajectory(None, None, torch.tensor(0.), None, None)\
            for _ in range(self.n_agents)
        ]
    
    def batch(self):
        '''Prioritized sampling of trajectories'''

        N = self.pos if not self.full else self.buffer_size

        # Normalized priorities
        P = self.priorities[:N] / self.priorities[:N].sum()

        # Selects `batch_size` trajectories with probabilities `P`
        idx = RNG.choice(N, size=self.batch_size, p=P.cpu().numpy())
        states = self.state_buffer[idx]
        actions = self.action_buffer[idx]
        rewards = self.reward_buffer[idx]
        last_states = self.last_state_buffer[idx]
        dones = self.done_buffer[idx]

        # Weights for model updates
        weights = ((N * P[idx])**(-self.beta)).unsqueeze(-1)

        # Save idx for priority updates
        self.last_idx = idx
  
        return states, actions, rewards, last_states, dones, weights

    def _append(self):
        '''Add current trajectory to buffer'''

        for exp in self.current_trajectory:
            self.state_buffer[self.pos] = exp.state
            self.action_buffer[self.pos] = exp.action
            self.reward_buffer[self.pos] = exp.reward
            self.last_state_buffer[self.pos] = exp.last_state
            self.done_buffer[self.pos] = exp.done
            self.priorities[self.pos] = self.max_priority
            self.pos = (self.pos + 1) % self.buffer_size
            if not self.pos:
                self.full = True

    def update_priorities(self, td_errors):
        '''
        Update priorites for records returned in last call to `batch`

        Parameters
        ----------
            td_errors : torch.tensor[float]
                `batch_size`-length tensor containing TD errors. Errors should correspond to data returned during last call to `self.batch`
        '''

        td_errors = (torch.abs(td_errors) + 1e-6)**0.75
        self.priorities[self.last_idx] = td_errors
        
        self.max_priority = torch.max(self.max_priority, td_errors.max())
        self.beta = min(self.eps * self.beta, 1.0)

    @property
    def ready(self):
        '''Flag indicating if the buffer contains enough records for at least one batch'''

        return self.full or (self.pos > self.batch_size)

    def __len__(self):
        '''Return the current size of buffer'''

        return len(self.buffer)

class MultiPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    '''Prioritized buffer for storing RL agent experience'''

    def __init__(self, state_size, action_size, trajectory_length, n_agents, gamma, shared=True, buffer_size=1e6, batch_size=64):
        '''
        Initialize a Memory object.

        Parameters
        ----------
            state_size : int
                Dimensionality of agent's state
            action_size : int
                Dimensionality of action space
            trajectory_length : int
                Number of steps to accumulate in each trajectory
            n_agents : int
                Number of agents adding to the buffer
            gamma : float
                Discount factor
            buffer_size : int
                Maximum number of records to hold in buffer
            batch_size : int
                Size of batches returned by Memory.batch
        '''

        super().__init__(state_size, action_size, trajectory_length, n_agents, gamma, buffer_size, batch_size)
        self.shared = shared
        # Containers for stored trajectories and associated priorities - track which agent produced each record
        self.state_buffer = torch.zeros((self.n_agents, self.buffer_size, self.state_size), device=device)
        self.action_buffer = torch.zeros((self.n_agents, self.buffer_size, self.action_size), device=device)
        self.reward_buffer = torch.zeros((self.n_agents, self.buffer_size), device=device)
        self.last_state_buffer = torch.zeros((self.n_agents, self.buffer_size, self.state_size), device=device)
        self.done_buffer = torch.zeros((self.n_agents, self.buffer_size), dtype=torch.bool, device=device)

        if self.shared:
            # Share priorities
            self.priorities = torch.zeros(self.buffer_size, device=device)
            self.last_idx = torch.zeros(self.batch_size, dtype=torch.long)
        else:
            self.priorities = torch.zeros((self.n_agents, self.buffer_size), device=device)
            self.last_idx = torch.zeros((self.n_agents, self.batch_size), dtype=torch.long)
    
    def batch(self, i=None):
        '''Prioritized sampling of trajectories'''

        assert self.shared or i is not None, "Must provide agent index when buffer is not shared"

        N = self.pos if not self.full else self.buffer_size

        # Normalized priorities
        if self.shared:
            P = self.priorities[:N] / self.priorities[:N].sum()
            idx = RNG.choice(N, size=self.batch_size, p=P.cpu().numpy())
        else:
            P = self.priorities[:,:N] / self.priorities[:,:N].sum(dim=1).unsqueeze(-1)
            idx = RNG.choice(N, size=self.batch_size, p=P[i].cpu().numpy())

        # Selects `batch_size` trajectories with probabilities `P`
        states = self.state_buffer[:, idx]
        actions = self.action_buffer[:, idx]
        rewards = self.reward_buffer[:, idx]
        last_states = self.last_state_buffer[:, idx]
        dones = self.done_buffer[:, idx]

        # Weights for model updates
        if self.shared:
            weights = ((N * P[idx])**(-self.beta)).unsqueeze(-1)
            self.last_idx[:] = torch.from_numpy(idx)
        else:
            weights = ((N * P[i, idx])**(-self.beta)).unsqueeze(-1)
            self.last_idx[i] = torch.from_numpy(idx)
  
        return states, actions, rewards, last_states, dones, weights

    def _append(self):
        '''Add current trajectory to buffer'''

        for i,exp in enumerate(self.current_trajectory):
            self.state_buffer[i, self.pos] = exp.state
            self.action_buffer[i, self.pos] = exp.action
            self.reward_buffer[i, self.pos] = exp.reward
            self.last_state_buffer[i, self.pos] = exp.last_state
            self.done_buffer[i, self.pos] = exp.done

        idx = self.pos if self.shared else (slice(None), self.pos)
        self.priorities[idx] = self.max_priority

        self.pos = (self.pos + 1) % self.buffer_size
        if not self.pos:
            self.full = True

    def update_priorities(self, td_errors, i=None):
        '''
        Update priorites for records returned in last call to `batch`

        Parameters
        ----------
            td_errors : torch.tensor[float]
                `batch_size`-length tensor containing TD errors. Errors should correspond to data returned during last call to `self.batch`
        '''

        assert self.shared or i is not None, "Must provide agent index when buffer is not shared"
        
        td_errors = (torch.abs(td_errors) + 1e-6)**0.75
        if self.shared:
            self.priorities[self.last_idx] = td_errors
        else:
            self.priorities[i, self.last_idx[i]] = td_errors
        
        self.max_priority = torch.max(self.max_priority, td_errors.max())
        self.beta = min(self.eps * self.beta, 1.0)
