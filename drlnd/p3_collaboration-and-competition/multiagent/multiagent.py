import pdb
import os

from multiagent import device, torch, optim, VMIN, VMAX, N_ATOMS, RNG
from multiagent.model import Actor, Critic, MultiCritic
from multiagent.buffer import PrioritizedReplayBuffer, MultiPrioritizedReplayBuffer
from multiagent.noise import GaussianNoise

class Agent:
    '''Single Distributional Actor-Critic Agent'''

    buffer = None

    def __init__(self,
        state_size, action_size, n_agents=None,
        vmin=-1.0, vmax=1.0, n_atoms=100,
        lr=0.001, gamma=0.99, trajectory_length=5,
        buffer_size=1e6, batch_size=256,
        actor_checkpoint=None, critic_checkpoint=None
    ):
        '''
        Initialize a MultiAgent.
        
        Parameters
        ----------
            state_size : int
                Dimensionality of agent's state
            action_size : int
                Dimensionality of action space
            n_agents : int
                Number of agents adding to the buffer
            vmin, vmax : float
                Upper & Lower bounds of the support for the critic model
            n_atoms : int
                Number of bins in the support
            lr : float
                Learning rate for NN updates
            gamma : float
                Discount factor
            trajectory_length : int
                Number of steps to accumulate in each trajectory
            buffer_size : int
                Maximum number of records to hold in replay buffer
            batch_size : int
                Size of batches returned by replay buffer
            actor_checkpoint : str or None
                Path to directory containing checkpoints of actor models
            critic_checkpoint : str or None
                Path to directory containing checkpoints of critic models
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.trajectory_length = trajectory_length
        self.gamma = gamma

        # Support for distributional critic
        self.vmin = vmin
        self.vmax = vmax
        self.n_atoms = n_atoms
        self.delta = (self.vmax - self.vmin) / (self.n_atoms - 1)
        self.atoms = torch.linspace(self.vmin, self.vmax, self.n_atoms, device=device).unsqueeze(0)

        self.actor_local = Actor(self.state_size, self.action_size).to(device)
        self.actor_target = Actor(self.state_size, self.action_size).to(device)
        if n_agents is None:
            self.critic_local = Critic(self.state_size, self.action_size, vmin=self.vmin, vmax=self.vmax, n_atoms=self.n_atoms).to(device)
            self.critic_target = Critic(self.state_size, self.action_size, vmin=self.vmin, vmax=self.vmax, n_atoms=self.n_atoms).to(device)
        else:
            self.critic_local = MultiCritic(self.state_size, self.action_size, n_agents, vmin=self.vmin, vmax=self.vmax, n_atoms=self.n_atoms).to(device)
            self.critic_target = MultiCritic(self.state_size, self.action_size, n_agents, vmin=self.vmin, vmax=self.vmax, n_atoms=self.n_atoms).to(device)

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

        if Agent.buffer is None:
            self.buffer = PrioritizedReplayBuffer(
                state_size=self.state_size, action_size=self.action_size,
                trajectory_length=self.trajectory_length, n_agents=1, gamma=self.gamma,
                buffer_size=buffer_size, batch_size=batch_size
            )

        self.noise = GaussianNoise(self.action_size, cov=0.1)

    def step(self, state, action, reward, next_state, done):
        '''
        Save experience in replay buffer. Return bool indicating readiness to update models

        Parameters
        ----------
            state : torch.tensor[float]
            action : torch.tensor[float]
            reward : torch.tensor[float]
            next_state : torch.tensor[float]
            done : torch.tensor[bool]
        '''

        return self.buffer.add(state, action, reward, next_state, done) and self.buffer.ready

    def act(self, state, train=False):
        '''
        Returns actions for given state as per current policy.
        
        Parameters
        ----------
            state : torch.tensor
                State of agent - returned from environmental observation
            explore : bool
                Whether to add noise to the chosen actions (facilitates exploration)
        '''

        with torch.no_grad():
            self.actor_local.eval()
            action = self.actor_local(state.to(device))
            self.actor_local.train()

        # Add noise for exploration
        if train:
            action += self.noise.sample()

        return torch.clamp(action, -1, 1)

    def update(self, tau=1e-3, soft_update=False):
        '''
        Update target model parameters.
        Hard update copies parameters; Soft update computes new values as: `θ_target = τ*θ_local + (1 - τ)*θ_target`

        Parameters
        ----------
            tau : float
                Interpolation parameter for soft updates. Ignored if `soft_update` is True
            soft_update : bool
                Use soft update
        '''

        for local_model, target_model in zip([self.critic_local, self.actor_local], [self.critic_target, self.actor_target]):
            if soft_update:
                for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                    target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            else:
                target_model.load_state_dict(local_model.state_dict())

    def project(self, next_distributions, rewards, dones):
        '''
        Projection of the sample Bellman update onto the support of the predicted distribution
            Based on Algorithm 1 from https://arxiv.org/pdf/1707.06887.pdf
            Adapted from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter14/06_train_d4pg.py

        For each value of `next_distributions` determine which bin(s) of `self.atoms` the value is mapped (projected) onto.
        `next_distributions` corresponds to the `next_action` take from `next_state` where `next_state` is obtained from the environment after an agent takes `action` from `state`.
        The objective here is to project the values in `next_distributions` onto the support of the distribution for `action`
        In the case of trajectories ending in a terminal state (ie. where `dones` == True), there is unit probability of achieving the corresponding values in rewards

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
        returns = torch.tile(rewards.unsqueeze(-1), (1, self.n_atoms))
        returns[~dones] += self.gamma**self.trajectory_length * self.atoms
        returns.clamp_(self.vmin, self.vmax)

        # values and corresponding upper/lower bounds of bins in the support
        b = (returns - self.vmin) / self.delta
        '''
        PRECISION HANDLING
        ==================
            In some cases, floating point errors cause values in `b` to equal (N_ATOMS-1) + 0.00000000000001
                e.g. for N_ATOMS = 100, values in `b` can sometimes equal 99.00000000000001
            When applying `ceil` to get upper bound on which bin the projection fits into, an index of 100 is returned
            This is out of range for a vector of length 100
            Reducing precision a little addresses this
        '''
        precision = 10
        b = torch.round(b * 10**precision) / 10**precision
        l, u = b.floor().long(), b.ceil().long()

        # split distribution values across bins, proportionate to distance from each bin edge
        ml_done, mu_done = (u - b), (b - l)
        ml = ml_done * next_distributions
        mu = mu_done * next_distributions

        # accumulate values into projection
        for i in range(next_distributions.size(0)):
            if dones[i]:
                projection[i].index_add_(0, l[i], ml_done[i])
                projection[i].index_add_(0, u[i], mu_done[i])
            else:
                projection[i].index_add_(0, l[i], ml[i])
                projection[i].index_add_(0, u[i], mu[i])

        # where values fall exactly into some bin, projection = next_distribution
        mask = l == u
        projection[mask] = next_distributions[mask]
        projection[dones][mask[dones]] = 1.0

        return projection

    def load(self, critic_checkpoint=None, actor_checkpoint=None, target=True, local=True):
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
            if target:
                self.critic_target.load_state_dict(torch.load(critic_checkpoint, map_location=device))
            if local:
                self.critic_local.load_state_dict(torch.load(critic_checkpoint, map_location=device))

        if actor_checkpoint is not None:
            print(f'Loading actor model weights from {actor_checkpoint}')
            if target:
                self.actor_target.load_state_dict(torch.load(actor_checkpoint, map_location=device))
            if local:
                self.actor_local.load_state_dict(torch.load(actor_checkpoint, map_location=device))

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

        if critic_checkpoint is not None:
            # print(f'Saving critic model weights to {critic_checkpoint}')
            torch.save(self.critic_target.state_dict(), critic_checkpoint)

        if actor_checkpoint is not None:
            # print(f'Saving actor model weights to {actor_checkpoint}')
            torch.save(self.actor_target.state_dict(), actor_checkpoint)

class MultiAgent:
    '''Pool of Distributional Actor-Critic Agents'''

    def __init__(self,
        state_size, action_size, n_agents,
        buffer_size=int(1e6), batch_size=128,
        gamma=0.99, lr=1e-3, tau=1e-3,
        trajectory_length=5, shared_buffer=True,
        actor_checkpoint=None, critic_checkpoint=None
    ):
        '''
        Initialize a MultiAgent
        
        Parameters
        ----------
            state_size : int
                Dimensionality of agent's state
            action_size : int
                Dimensionality of action space
            n_agents : int
                Number of agents adding to the buffer
            buffer_size : int
                Maximum number of records to hold in replay buffer
            batch_size : int
                Size of batches returned by replay buffer
            gamma : float
                Discount factor
            lr : float
                Learning rate for NN updates
            tau : float
                Interpolation factor for soft updates of target networks
            trajectory_length : int
                Number of steps to accumulate in each trajectory
            shared_buffer : bool
                Have agents collect experience in a single, shared buffer
            actor_checkpoint : str or None
                Path to directory containing checkpoints of actor models
            critic_checkpoint : str or None
                Path to directory containing checkpoints of critic models
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length
        self.tau = tau
        self.gamma = gamma
        self.shared_buffer = shared_buffer

        # Params for distributional critic support
        self.vmin = VMIN
        self.vmax = VMAX * trajectory_length
        self.n_atoms = N_ATOMS

        Agent.buffer = MultiPrioritizedReplayBuffer(
            state_size=self.state_size, action_size=self.action_size,
            trajectory_length=self.trajectory_length, n_agents=self.n_agents, gamma=self.gamma,
            buffer_size=buffer_size, batch_size=batch_size, shared=self.shared_buffer
        )

        # Instantiate agents
        actor_checkpoints = os.scandir(actor_checkpoint) if actor_checkpoint is not None else None
        critic_checkpoints = os.scandir(critic_checkpoint) if critic_checkpoint is not None else None
        self.agents = list()
        for i in range(self.n_agents):
            chkpt_a = None if actor_checkpoints is None else next(actor_checkpoints).path
            chkpt_c = None if critic_checkpoints is None else next(critic_checkpoints).path
            self.agents.append(
                Agent(
                    state_size=self.state_size, action_size=self.action_size, n_agents=self.n_agents,
                    vmin=self.vmin, vmax=self.vmax, n_atoms=self.n_atoms,
                    lr=lr, gamma=self.gamma, trajectory_length=self.trajectory_length,
                    buffer_size=buffer_size, batch_size=batch_size,
                    actor_checkpoint=chkpt_a, critic_checkpoint=chkpt_c
                )
            )

        self.agents = tuple(self.agents)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        '''
        Save experience in replay buffer. Return bool indicating readiness to update models

        Parameters
        ----------
            states : torch.tensor[float]
            actions : torch.tensor[float]
            rewards : torch.tensor[float]
            next_states : torch.tensor[float]
            dones : torch.tensor[bool]
        '''

        return Agent.buffer.add(states, actions, rewards, next_states, dones) and Agent.buffer.ready

    def act(self, states, train=False):
        '''
        Returns actions for given state as per current policy.
        
        Parameters
        ----------
            states : List[np.ndarray]
            train : bool
                Toggle training behavior. Training will start with `explore_steps` number of random actions; thereafter, will add noise to all model-generated actions
        '''

        if train and (self.t_step < 1000):
            # Frist 1K steps of training are random actions
            self.t_step += 1
            return torch.from_numpy(RNG.uniform(-1,1, size=(self.n_agents, self.action_size)))

        actions = torch.zeros((self.n_agents, self.action_size), device=device)
        for i, (agent,state) in enumerate(zip(self.agents, states)):
            actions[i] = agent.act(states[i].unsqueeze(0), train=train)

        return actions.cpu()

    def update(self, soft_update=False):
        '''
        Update target model parameters. Hard update copies parameters; Soft update computes new values as:  ```θ_target = τ*θ_local + (1 - τ)*θ_target```

        Parameters
        ----------
            soft_update : bool
                Flag for running a soft update. If `False` copies params exactly
        '''

        for agent in self.agents:
            agent.update(tau=self.tau, soft_update=soft_update)

    def learn(self):
        '''Update local models with batch of experience from replay buffer'''

        ####################################################################################
        # `batch` returns tensors with dims (n_agents x batch_size x N) where N is state_size, etc.
        # `all_states`, etc. have dims (n_agents x n_agents x batch_size x N)
        # Last dimension is omitted from rewards, dones, and weights
        ####################################################################################
        all_states, all_actions, all_rewards, all_last_states, all_dones, all_weights = \
            [torch.stack(x) for x in zip(*[agent.buffer.batch(i) for i,agent in enumerate(self.agents)])]

        for i, (agent, states, actions, rewards, last_states, dones, weights) in\
            enumerate(zip(self.agents, all_states, all_actions, all_rewards, all_last_states, all_dones, all_weights)):

            idx = [j for j in range(self.n_agents) if j!=i]
            rewards = rewards[i]
            dones = dones[i]

            # ------------------- update critic ------------------- #

            with torch.no_grad():
                # All agents' end-state predictions
                last_actions = torch.stack([_agent_.actor_target(last_states[j]) for j, _agent_ in enumerate(self.agents)])
                # Other agents' end-state predictions
                other_last_actions = self.reshape(last_actions[idx])
                # End-state value distribution
                last_distributions = agent.critic_target(self.reshape(last_states), last_actions[i], other_last_actions)
                # Project realized rewards on support
                targets = agent.project(last_distributions, rewards, dones)

            # Other actors decisions
            other_actions = self.reshape(actions[idx])
            # Starting state's predicted value distribution
            distributions = agent.critic_local(self.reshape(states), actions[i], other_actions, log_prob=True)
            # cross entropy loss => -∑(p * log(q))
            td = -(weights * targets * distributions).sum(-1)
            critic_loss = td.mean()

            # backprop
            agent.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.critic_optim.step()

            # ------------------- update actor ------------------- #

            # All actors predictions
            actions_pred = [_agent_.actor_target(states[j]) for j, _agent_ in enumerate(self.agents)]
            # Other actors predictions
            other_actions_pred = self.reshape(torch.stack([x.detach() for j,x in enumerate(actions_pred) if j!=i]))

            # Predicted value of state/action pair = E[value_distribution] = ∑ p_i * i
            action_probs = agent.critic_local(self.reshape(states), actions_pred[i], other_actions_pred)
            action_values = (action_probs * agent.atoms).sum(-1)
            actor_loss = -action_values.mean()

            # backprop
            agent.actor_optim.zero_grad()
            actor_loss.backward()
            agent.actor_optim.step()

            # ------------------- update priorities ------------------- #

            agent.buffer.update_priorities(td.detach(), i)

    def reshape(self, x):
        '''
        Reshape torch.tensor for forward pass of a MultiCritic model

        Parameters
        ----------
            x : torch.tensor
                Input tensor to MultiCritic.forward (`states` or `other_actions`)
                Dimensions (n_agents x batch_size x <state/action>_size) are output as (batch_size x (n_agents * <state/action>_size))
        '''

        return x.transpose(0,1).flatten(start_dim=1)

    def load(self, critic_checkpoint_dir=None, actor_checkpoint_dir=None, target=True, local=True):
        '''
        Load model weights from checkpoint file

        Parameters
        ----------
            actor_checkpoint : str or None
                Path to file containing state_dict of actor model
            critic_checkpoint : str or None
                Path to file containing state_dict of critic model
        '''

        if critic_checkpoint_dir is not None:
            print(f'Loading critic model weights from {critic_checkpoint_dir}')
            for agent, critic_checkpoint in zip(self.agents, os.scandir(critic_checkpoint_dir)):
                agent.load(critic_checkpoint=critic_checkpoint.path, target=target, local=local)

        if actor_checkpoint_dir is not None:
            print(f'Loading actor model weights from {actor_checkpoint_dir}')
            for agent, actor_checkpoint in zip(self.agents, os.scandir(actor_checkpoint_dir)):
                agent.load(actor_checkpoint=actor_checkpoint.path, target=target, local=local)

    def save(self, actor_checkpoint_dir=None, critic_checkpoint_dir=None, score=None):
        '''
        Save models' weights to checkpoint files

        Parameters
        ----------
            actor_checkpoint : str or None
                Path to file to save state_dict of actor model
            critic_checkpoint : str or None
                Path to file to save state_dict of critic model
        '''


        if critic_checkpoint_dir is not None:
            # print(f'Saving agents\' critic models to {critic_checkpoint_dir}')
            for i in range(self.n_agents):
                c = f'critic{i}_checkpoint.pth' if score is None else f'critic{i}_checkpoint_{score:.6f}.pth'
                critic_checkpoint = os.path.join(critic_checkpoint_dir, c)
                self.agents[i].save(critic_checkpoint=critic_checkpoint)

        if actor_checkpoint_dir is not None:
            # print(f'Saving actor model weights to {actor_checkpoint_dir}')
            for i in range(self.n_agents):
                a = f'actor{i}_checkpoint.pth' if score is None else f'actor{i}_checkpoint_{score:.6f}.pth'
                actor_checkpoint = os.path.join(actor_checkpoint_dir, a)
                self.agents[i].save(actor_checkpoint=actor_checkpoint)