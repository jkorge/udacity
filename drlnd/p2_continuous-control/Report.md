# Project 2 - Control

This repo contains a submission for the second project of Udacity's Deep Reinforcement Learning Nanodegree. This submission implements a D4PG Agent as a means to solving the provided Reacher environment.

## Requirements

The task at had is to keep one end of a double-jointed arm within a move target region. The player receives a reward of +0.1 for every timestep the arm is successfully held in the target region. The player is deemd to have solved the environment upon attaining an average score of +30 over a continuous window of 100 episodes of play.

## Methods

### Distributed Distributional Deep Deterministic Policy Gradients (D4PG)

At the heart of this program is a Reinforcement Learning (RL) agent designed to control the arm and learn from its experience using an Actor-Critic framework known as Distributed Distributional Deep Deterministic Policy Gradients (D4PG). 

The actor component is a neural network which produces a prediction of the optimal action the agent should take at each timestep - an approximation of the optimal policy π<sup>\*</sup>. The critic, another neural network, predicts a distribution of state-action values - an approximation of the distribution over the range of the Q-value function. In this implementation the distribution is a categorical distribution over a predefined support. Each value of the distribution represents the probability that the action, provided by the actor network, will result in that value. For this implementation the support is defined over the range `[0, 0.5]` since the agent will only every receive at most 0.5 over a 5-step trajectory. The number of bins within the support is set to 100.

#### Updates

As in the first project, the models are updated in a manner befitting deep Q-learning. Two instances of each network are instantiated - a local and a target network. The former network predicts what actions the agent should take and their corresponding values; the latter, the target values against which the local model will be compared.

At each timestep the local models are provided four minibatches of data and update their parameters after computing the loss of each batch. Then, the target networks are updated with an exact copy of the local networks' parameters. Soft updates were tested but did not demonstrate improvements to the networks' performance.

### Experience Replay

In the simplest case Q-Learning will have the agent update its network parameters at every timestep. However, consecutive actions may be strongly correlated and can impair the network's learning process when these updates are applied consecutively. To mediate this instability Deep Q-Learning will have the agent collect tuples of state-action-reward transitions and sample these randomly to generate a minibatch of data for updating the neural network.  
In this implementation, the 20-agent environment formed the basis for all development. In this context, experiences are futher decoupled by the fact that the networks are updated from experiences sampled randomly from a buffer to which multiple agents contribute.

#### N-Step Bootstrapping

Somewhere between Monte-Carlo learning and TD-Learning lies N-Step Bootstrapping. In this variation, rather than learning from an entire episode of experience (as in Monte-Carlo) or from a single timestep of experience (as in TD learning) the agent learns after some intermediate value of timesteps. In this implementation the default value of 5-step trajectories were stored to the replay buffer. The saved data from the trajectories comprised the state of the agent at the start of the trajectory, the action the agent took from that state, the total discounted return of the entire trajectory, the state of the agent at the end of the trajectory, and a boolean flag indicating whether the agent arrived at a terminal state. The advantage here is to balance uncertainty with timely convergence.

#### Prioritization

Not all experience is equal. Trajectories which yield a larger error are more valuable to the learning process. To take advantage of this, the replay buffer maintains a record of each trajectories priority which can be used to preferentially select those trajectories. This priority is, in practice, the TD error of the critic model's prediction. These priorities, after normalization, define a probablitity of drawing each corresponding unit of experience for the next minibatch of training data. Accounting for this requires modifying the updates to the critic model by a value inversely proportional to the size of the replay buffer and the priorities of each record therein:  
![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;(\frac{1}{N&space;*&space;P_{i}})^\beta)

### Network Architecture

The actor network is comprised of two fully connected layers with a ReLU activation in between finishing with a tanh activation (range of tanh is `[-1,1]` as required by the environment). The critic network has four fully connected layers and two inputs. The first fully connected layer produces an encoding of the state vector. This encoding is concatenated with the action vector and passed through the rest of the network. Leaky ReLU activations are applied between the fully connected layers. The output is either filtered through a softmax or a log-softmax depending on the context. Specifically, the log of the probability vector is required for outputs from the local network in order to compute the cross-entropy with the output of the target network; this cross-entropy is the loss for the local network.

Actor:

| Layer    | Output Shape | Param # |
| ---------| ------------ | ------- |
| Input    | [256, 33]    | 8,448   |
| Linear-1 | [256, 256]   | 8,704   |
| Linear-2 | [256, 4]     | 1,028   |

Critic:

| Layer    | Output Shape | Param # |
| ---------| ------------ | ------- |
| Input-1  | [256, 33]    | 8,448   |
| Linear-1 | [256, 256]   | 8,704   |
| Input-2  | [256, 4]     | 1,024   |
| Linear-2 | [256, 256]   | 1,028   |
| Linear-3 | [256, 128]   | 1,028   |
| Linear-4 | [256, 100]   | 1,028   |

### Hyperparameter Selection

A few learning rates were tested (0.1, 0.01, 0.001, 0.00001) but only with an initial learning rate of 0.0001 did the models demonstrate meaningful progress. Other learning rates either lead to no learning or convergence to a suboptimal model. Trajectory lengths ranging from 1 to 1000 were examined, but the traditionally value of 5 show stable learning.

| Parameter            | Value   |
| -------------------- | ------- |
| Learning Rate        | 0.0001  |
| Discount Factor      | 0.99    |
| Batch Size           | 256     |
| Buffer Size          | 1000000 |
| Trajectory Length    | 5       |

Models were trained with an Adam optimizer. A few others were tested (AMSGrad, SGD with Nesterov Momentum) to no avail. Interestingly, the tracking of the maximum observed gradient by AMSGrad helped early in the learning process but seemed to converge to a subobtimal network (possibly since prioritized experience replay already accounts for the most informative records) and dramatically slowed the learning process.

## Results

The agent was able to solve the environment in 108 episodes using the hyperparameters from the preceding table and an Adam optimizer. The 100-episode average in the graph below is computed by tracking the score each of the 20 agents achieves in each episode, averaging those, accumulating them in a deque of length 100, and taking the average over the values of that deque. It is an average of the average score achieved by all agents over the previous 100 episodes.  
![Training Scores](/media/training_scores.jpg)

## Next Steps

Futher experimentation is necessary to ensure that the chosen hyperparamters are indeed optimal. Permutations of different values, training the networks with different optimizers and loss functions, and learning rate decay are all methods which can be incorporated into a automated training pipeline which tests and reports the scores attained from different permutations of values and algorithms. Once these values are set, a few amendments to the learning process may be made to address other factors.

This method was originally proposed with mention made of using a critic model which predicts the parameters of a Gaussian mixture model. It remains to be seen whether this method, very similar to the reparameterization method of Variational Autoencoders, would yield meaningful results in this problem space. Futhermore, under the Categorial distributional predictions implemented here, no experimental results were obtained to measure the influence of varying the range and number of bins within the underlying support. In theory more bins would improve performance as the results can better align with achievable values. This is due to the fact that outputs from the target model have to be projected back onto the support since the stochastic nature of the model does not guarantee its output values will lie in the range of the support. The process of projecting the predicted distribution requires splitting values into nearby bins in proportion with their distance from that bin.

## Sources

There are a few sources which were pivotal to completing this project. First, there's the [original D4PG report](https://arxiv.org/pdf/1804.08617.pdf) and [the paper which lead to its development](https://arxiv.org/pdf/1707.06887.pdf). The code which implements the projection of the target distribution on the support of the local distribution was adapted from [code](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter14/06_train_d4pg.py) to accompany the book "Deep Reinforcement Learning Hands On" by Maxim Lapan. A [few ideas](https://knowledge.udacity.com/questions/37819) presented in the Udacity Knowledge by a previous student who also attempted D4PG were helpful. Although most didn't end up in this submission, testing them helped to bring about a better understanding of the environment and how to adapt this esoteric RL mechanism to it.