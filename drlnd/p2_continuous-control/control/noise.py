import random
import copy

from control import torch, MultivariateNormal, np, RNG, device

class OUNoise:
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        '''Initialize parameters and noise process.'''

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu).'''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''Update internal state and return it as a noise sample.'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([RNG.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class GaussianNoise:
    '''Normal Distribution'''

    def __init__(self, size, mu=0., cov=1.):
        '''Initialize parameters'''

        mu = mu + torch.zeros(size, device=device)
        cov = cov * torch.eye(size, device=device)
        self.distrib = MultivariateNormal(mu, covariance_matrix=cov)
        self.eps = 0.3

    def sample(self, n=1):
        '''Draw n samples from a Gaussian distribution'''

        return self.distrib.sample((n,))

    def decay(self):
        '''Reduce variance of underlying distribution'''

        self.eps = max(0.005, 0.95 * self.eps)
