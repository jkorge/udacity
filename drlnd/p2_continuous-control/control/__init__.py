import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
default_dtype = torch.float64 if (sys.maxsize > 2**32) else torch.float32
torch.set_default_dtype(default_dtype)
torch.manual_seed(0)
RNG = np.random.default_rng(0)