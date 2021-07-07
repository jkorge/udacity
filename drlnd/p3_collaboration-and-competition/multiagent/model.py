from multiagent import torch, nn, F

class Actor(nn.Module):
    '''Deterministic Policy Model'''

    def __init__(self, state_size, action_size):
        '''
        Initialize parameters and build model.
        Parameters
        ----------
            state_size : int
                Dimension of each state
        '''

        super().__init__()

        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        '''
        Forward pass

        Parameters
        ----------
            state : torch.tensor[Float]
                Batch of agent's oberservations
        '''

        x = self.bn(state)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

class Critic(nn.Module):
    '''Distributional Value Model'''

    def __init__(self, state_size, action_size, vmin=-1.0, vmax=1.0, n_atoms=51):
        '''
        Initialize parameters and build model.
        Parameters
        ----------
            state_size : int
                Dimension of each state
            action_size : int
                Dimension of each action
            vmin, vmax : float
                Upper & Lower bounds of the support
            n_atoms : int
                Number of bins in the support
        '''

        super().__init__()

        # Create non-trainable member tensor
        self.register_buffer('support', torch.linspace(vmin, vmax, n_atoms))

        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_atoms) # predict probs over each bin in support

    def forward(self, states, actions, log_prob=False):
        '''
        Forward pass

        Parameters
        ----------
            state : torch.tensor[Float]
                Batch of agents' oberservations. 2D tensor w/ dims (batch_size x (state_dim * n_agents))
            action : torch.tensor[Float]
                Batch of agents' actions. 2D tensor w/ dims (batch_size x (action_dim * n_agents))
            log_prob : bool
                Flag for returning the log of the predicted probabilities (instead of the probabilities)
        '''

        x = self.bn(states)
        x = F.leaky_relu(self.fc1(x))
        x = torch.cat([x, actions], dim=-1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        if log_prob:
            return F.log_softmax(x, dim=-1)
        else:
            return F.softmax(x, dim=-1)

class MultiCritic(nn.Module):
    '''Multi-Agent Distributional Value Model'''

    def __init__(self, state_size, action_size, n_agents, vmin=-1.0, vmax=1.0, n_atoms=51):
        '''
        Initialize parameters and build model.
        Parameters
        ----------
            state_size : int
                Dimension of each state
            action_size : int
                Dimension of each action
            n_agents : int
                Total number of agents
            vmin, vmax : float
                Upper & Lower bounds of the support
            n_atoms : int
                Number of bins in the support
        '''

        super().__init__()

        # Create non-trainable member tensor
        self.register_buffer('support', torch.linspace(vmin, vmax, n_atoms))

        input_size = (n_agents * state_size) + ((n_agents-1) * action_size)
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_atoms) # predict probs over each bin in support

    def forward(self, states, actions, other_actions, log_prob=False):
        '''
        Forward pass

        Parameters
        ----------
            state : torch.tensor[Float]
                Batch of agents' oberservations. 2D tensor w/ dims (batch_size x (state_dim * n_agents))
            action : torch.tensor[Float]
                Batch of agents' actions. 2D tensor w/ dims (batch_size x (action_dim * n_agents))
            log_prob : bool
                Flag for returning the log of the predicted probabilities (instead of the probabilities)
        '''

        x = torch.cat([states, other_actions], dim=-1)
        x = self.bn(x)
        x = F.leaky_relu(self.fc1(x))
        x = torch.cat([x, actions], dim=-1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        if log_prob:
            return F.log_softmax(x, dim=-1)
        else:
            return F.softmax(x, dim=-1)
