import random
import copy

from multiagent import torch, MultivariateNormal, np, RNG, device

class OUNoise:
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        '''Initialize parameters and noise process.'''

        self.size = size
        self.mu = mu * torch.ones(size, device=device)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu).'''

        self.state = copy.deepcopy(self.mu)

    def sample(self):
        '''Update internal state and return it as a noise sample.'''

        self.state += self.theta * (self.mu - self.state) + self.sigma * torch.from_numpy(RNG.random(self.size)).to(device)
        return self.state

class GaussianNoise:
    '''Normal Distribution'''

    def __init__(self, size, mu=0., cov=1., eps=0.3):
        '''
        Instantiate Gaussian Noise object

        Parameters
        ----------
            size : int
                Number of dimensions of the underlying distribution
            mu : float or torch.tensor[float]
                Mean value of each dimension. If `float`, each dimension will have the same mean
            cov : float
                Value used to construct diagonal covariance tensor
            eps : float
                Multipier for samples drawn from underlying distribution
        '''

        mu = mu + torch.zeros(size, device=device)
        cov = cov * torch.eye(size, device=device)
        self.eps = eps
        self.distrib = MultivariateNormal(mu, covariance_matrix=cov)

    def sample(self, n=1):
        '''Draw n samples from a Gaussian distribution'''

        return self.eps * self.distrib.sample((n,))