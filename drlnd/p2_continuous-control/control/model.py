from control import torch, nn, F

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
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        '''
        Forward pass

        Parameters
        ----------
            state : torch.tensor[Float]
                Batch of agents' oberservations
        '''

        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))

class Critic(nn.Module):
    '''Distributional Value Model'''

    def __init__(self, state_size, action_size, vmin, vmax, n_atoms=51):
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

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_atoms) # predict probs over each bin in support

    def forward(self, state, action, log_prob=False):
        '''
        Forward pass

        Parameters
        ----------
            state : torch.tensor[Float]
                Batch of agents' oberservations
            action : torch.tensor[Float]
                Batch of agents' actions
            log_prob : bool
                Flag for returning the log of the predicted probabilities (instead of the probabilities)
        '''

        x = F.leaky_relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        if log_prob:
            return F.log_softmax(x, dim=-1)
        else:
            return F.softmax(x, dim=-1)