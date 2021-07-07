import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

# Use GPU where available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prefer 64-bit
default_dtype = torch.float64 if (sys.maxsize > 2**32) else torch.float32
torch.set_default_dtype(default_dtype)

# Random seeds
torch.manual_seed(0)
RNG = np.random.default_rng(0)

# Parameters of the support vector for the Critc's predicted distribution
VMIN = -0.01    # Min 1-step reward of -0.01
VMAX =  0.10    # Max 1-step reward of +0.10
N_ATOMS = 100