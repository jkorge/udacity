import os

from control import device, RNG, torch, np, optim
from control.model import Actor, Critic
from control.memory import PrioritizedReplayBuffer
from control.noise import OUNoise, GaussianNoise

# Parameters of the support vector for the Critc's predicted distribution
VMIN = 0        # Negative rewards not given by env
VMAX = 0.1      # Max 1-step reward of +0.1
N_ATOMS = 100
DELTA = (VMAX - VMIN) / (N_ATOMS - 1)
ATOMS = torch.linspace(VMIN, VMAX, N_ATOMS).to(device)

class Agent():
    '''RL Agent Implementing Distributional Deep Determinisitc Policy Gradient'''

    def __init__(self, state_size, action_size, n_agents,
                    buffer_size=int(1e6), batch_size=128,
                    gamma=0.99, lr=1e-3, trajectory_length=5,
                    actor_checkpoint=None, critic_checkpoint=None):
        '''
        Initialize a D4PG Agent.
        
        Parameters
        ----------
            state_size : int
                Dimensionality of agent's state
            action_size : int
                Dimensionality of action space
            n_agents : int
                Number of agents adding to the buffer
            trajectory_length : int
                Number of steps to accumulate in each trajectory
            buffer_size : int
                Maximum number of records to hold in replay buffer
            batch_size : int
                Size of batches returned by replay buffer
            update_freq : int
                Number of trajectories to collect before NN training iteration
            gamma : float
                Discount factor
            lr : float
                Learning rate for NN updates
            vmin, vmax : float, float
                Min/Max values for the support of Critic model's predicted distribution
            n_atoms : int
                Number of bins in the support of Critic model's predicted distribution
            actor_checkpoint : str or None
                Path to file containing state_dict of actor model
            critic_checkpoint : str or None
                Path to file containing state_dict of critic model
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.trajectory_length = trajectory_length
        self.n_agents = n_agents

        # update VMAX, etc. to account for N-step learning
        global VMAX, DELTA, ATOMS
        VMAX *= self.trajectory_length
        DELTA = (VMAX - VMIN) / (N_ATOMS - 1)
        ATOMS = torch.linspace(VMIN, VMAX, N_ATOMS).to(device)

        # Actor-Critic networks
        self.actor_target = Actor(self.state_size, self.action_size).to(device)
        self.actor_local = Actor(self.state_size, self.action_size).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, VMIN, VMAX, N_ATOMS).to(device)
        self.critic_local = Critic(self.state_size, self.action_size, VMIN, VMAX, N_ATOMS).to(device)

        # Load checkpoints to target networks
        if (actor_checkpoint is not None) and not os.path.exists(actor_checkpoint):
            actor_checkpoint = None
        if (critic_checkpoint is not None) and not os.path.exists(critic_checkpoint):
            critic_checkpoint = None
        if (critic_checkpoint is not None) or (actor_checkpoint is not None):
            self.load(critic_checkpoint, actor_checkpoint)

        # Synchronize local and target networks
        self.actor_local.load_state_dict(self.actor_target.state_dict())
        self.critic_local.load_state_dict(self.critic_target.state_dict())

        # Optimizers
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=lr)

        # Discount/interplation factors
        self.gamma = gamma

        # Replay memory
        self.buffer = PrioritizedReplayBuffer(
            state_size=self.state_size, action_size=self.action_size,
            trajectory_length=self.trajectory_length, n_agents=n_agents, gamma=self.gamma,
            buffer_size=buffer_size, batch_size=batch_size
        )

        # Process noise
        # self.noise = OUNoise(action_size)
        self.noise = GaussianNoise(action_size)

        self.t_step = 0
        self.N = 4
    
    def step(self, states, actions, rewards, next_states, dones):
        '''Save experience in replay buffer. Return bool indicating readiness to update models'''

        return self.buffer.add(states, actions, rewards, next_states, dones) and self.buffer.ready

    def act(self, states, explore=False):
        '''
        Returns actions for given state as per current policy.
        
        Parameters
        ----------
            states : List[np.ndarray]
            explore : bool
                Whether to add noise to the chosen actions (facilitates exploration)
        '''

        with torch.no_grad():
            actions = self.actor_local(states.to(device)).detach()

        # Add noise for exploration
        if explore:
            actions += self.noise.sample(self.n_agents)

        return torch.clip(actions, -1,1).cpu()

    def learn(self):
        '''Update local models with batch of experience from replay buffer'''

        states, actions, rewards, last_states, dones, weights = self.buffer.batch()

        # ------------------- update critic ------------------- #
        
        # Last actions and value distributions
        last_actions = self.actor_target(last_states)
        last_distributions = self.critic_target(last_states, last_actions)

        # project last distributions back onto support
        targets = self.project(last_distributions, rewards).detach()

        # starting state's value distribution
        distributions = self.critic_local(states, actions, log_prob=True)

        # cross entropy loss => -∑(p * log(q))
        td = -(weights * targets * distributions).sum(-1)
        critic_loss = td.mean()

        # backprop
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        # ------------------- update actor ------------------- #

        # starting state predicted actions
        actions_pred = self.actor_local(states)

        # predicted value of state/action pair
        action_probs = self.critic_local(states, actions_pred)
        action_values = (action_probs * ATOMS.unsqueeze(0)).sum(-1)
        actor_loss = -action_values.mean()

        # backprop
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ------------------- update priorities ------------------- #

        self.buffer.update_priorities(td.detach())

    def update(self, tau=1e-3, soft_update=False):
        '''
        Update target model parameters. Hard update copies parameters; Soft update computes new values as:  
        ```
            θ_target = τ*θ_local + (1 - τ)*θ_target
        ```

        Parameters
        ----------
            local_model : torch.nn.Module
                Copy weights FROM this model
            target_model : torch.nn.Module
                Copy weights TO this model
            tau : float
                interpolation parameter for soft updates
        '''

        for local_model, target_model in zip([self.critic_local, self.actor_local], [self.critic_target, self.actor_target]):
            if soft_update:
                for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                    target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            else:
                target_model.load_state_dict(local_model.state_dict())

    def load(self, critic_checkpoint=None, actor_checkpoint=None):
        '''
        Load model weights from checkpoint file

        Parameters
        ----------
            actor_checkpoint : str or None
                Path to file containing state_dict of actor model
            critic_checkpoint : str or None
                Path to file containing state_dict of critic model
        '''

        if critic_checkpoint is not None:
            print(f'Loading critic model weights from {critic_checkpoint}')
            self.critic_target.load_state_dict(torch.load(critic_checkpoint, map_location=device))

        if actor_checkpoint is not None:
            print(f'Loading actor model weights from {actor_checkpoint}')
            self.actor_target.load_state_dict(torch.load(actor_checkpoint, map_location=device))


    def save(self, actor_checkpoint=None, critic_checkpoint=None):
        '''
        Save models' weights to checkpoint files

        Parameters
        ----------
            actor_checkpoint : str or None
                Path to file to save state_dict of actor model
            critic_checkpoint : str or None
                Path to file to save state_dict of critic model
        '''

        if critic_checkpoint is None:
            critic_checkpoint = 'checkpoints/critic_checkpoint.pth'
        if actor_checkpoint is None:
            actor_checkpoint = 'checkpoints/actor_checkpoint.pth'

        print(f'Saving critic model weights to {critic_checkpoint}')
        torch.save(self.critic_target.state_dict(), critic_checkpoint)

        print(f'Saving actor model weights to {actor_checkpoint}')
        torch.save(self.actor_target.state_dict(), actor_checkpoint)

    def project(self, next_distributions, rewards):
        '''
        Projection of the sample Bellman update onto the support of the predicted distribution
            Based on Algorithm 1 from https://arxiv.org/pdf/1707.06887.pdf
            Adapted from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter14/06_train_d4pg.py

        For each value of `next_distributions` determine which bin(s) of `ATOMS` the value is mapped (projected) onto.
        `next_distributions` corresponds to the `next_action` take from `next_state` where `next_state` is obtained from the environment after an agent takes `action` from `state`.
        The objective here is to project the values in `next_distributions` onto the support of the distribution for `action`.

        Parameters
        ----------
            next_distributions : torch.tensor[Float]
                Raw predictions (return value of Critic.forward)
            rewards : torch.tensor[Float]
                Discounted rewards obtained from the environment over an N-step trajectory
            dones : torch.tensor[Bool]
                Flag indicating which records correspond to terminal states
            gamma : float
                Discount factor
        '''

        # allocate projection array
        projection = torch.zeros(next_distributions.size()).to(device)

        # contraction of support by gamma, shifted by rewards
        # analogous to discounted future returns in the usual bellman operator
        returns = rewards.unsqueeze(-1) + self.gamma**self.trajectory_length * ATOMS.unsqueeze(0)
        returns.clamp_(VMIN, VMAX)

        # values and corresponding upper/lower bounds of bins in the support
        b = (returns - VMIN) / DELTA
        l, u = b.floor().long(), b.ceil().long()

        # split distribution values across bins, proportionate to distance from each bin edge
        ml = ((u - b) * next_distributions)
        mu = ((b - l) * next_distributions)

        # accumulate values into projection
        for i in range(next_distributions.size(0)):
            projection[i].index_add_(0, l[i], ml[i])
            projection[i].index_add_(0, u[i], mu[i])

        # where values fall exactly into some bin, projection = next_distribution
        mask = l == u
        projection[mask] = next_distributions[mask]


        return projection