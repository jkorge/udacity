# Project 1 - Navigation

This repo contains a submission for the first project of Udacity's Deep Reinforcement Learning Nanodegree. This submission implements a Deep Q-Learning Agent as a means to solving the provided Banana Collection environment.

## Requirements

The task at had is to navigate a bounded plane with a random arrangement of yellow and blue bananas while trying to collect the yellows and avoid the blues. Each yellow banana earns a reward of +1; blue bananas, -1. The player is deemd to have solved the environment upon attaining an average score of +13 over a continuous window of 100 episodes of play.

## Methods

### Deep Q-Learning

At the heart of this program is a Reinforcement Learning (RL) agent designed to navigate the environment and learn from its experience using Deep Q-Learning.

The objective in Q-Learning is to learn the function which yields the correct value of the reward an agent will receive when taking an action from its current state. In Deep Q-Learning this function is approximated by a deep neural network. Deep Q-Learning also introduces two important concepts: experience replay and fixed Q-targets.

#### Experience Replay

In the simplest case Q-Learning will have the agent update its network parameters at every timestep. However, consecutive actions may be strongly correlated and can impair the network's learning process when these updates are applied consecutively. To mediate this instability Deep Q-Learning will have the agent collect tuples of state-action-reward transitions and sample these randomly to generate a minibatch of data for updating the neural network.

#### Fixed Q-Targets

When updating the parameters of the network another source of undesired correlation lies within the computation of the TD Error. This error term is the difference between the predicted Q-value of the origin state and action in a transition and the maximum predicted Q-value of the successor state (discounted and added to the immediate reward). The error functions as the learning rate (or a product of it) for the update of the neural network parameters.

![equation](https://latex.codecogs.com/gif.latex?%5CDelta%20w%20%3D%20%5Calpha%20%5Ccdot%20%5Coverbrace%7B%28%20%5Cunderbrace%7BR%20&plus;%20%5Cgamma%20%5Cmax_a%5Chat%7Bq%7D%28S%27%2C%20a%2C%20w%29%7D_%7B%5Crm%20%7BTD%7Etarget%7D%7D%20-%20%5Cunderbrace%7B%5Chat%7Bq%7D%28S%2C%20A%2C%20w%29%7D_%7B%5Crm%20%7Bold%7Evalue%7D%7D%29%7D%5E%7B%5Crm%20%7BTD%7Eerror%7D%7D%20%5Cnabla_w%5Chat%7Bq%7D%28S%2C%20A%2C%20w%29)

Since the TD target and the origin-state/action prediction are using the same neural network, the network weights adjust to more closely match the target by comparing itself to the target. This correlation between the network and the target causes training to oscillate.  
To address this issue, two neural networks are maintained: a primary (local) network  and a secondary (target) network. These networks have identical architectures. The target network, however, lags behind the local network. Only after a certain number of online updates to the local network will the target network copy the parameters from the local network.

![equation](https://latex.codecogs.com/gif.latex?%5CDelta%20w%20%3D%20%5Calpha%20%5Ccdot%20%5Coverbrace%7B%28%20%5Cunderbrace%7BR%20&plus;%20%5Cgamma%20%5Cmax_a%5Chat%7Bq%7D%28S%27%2C%20a%2C%20w%5E-%29%7D_%7B%5Crm%20%7BTD%7Etarget%7D%7D%20-%20%5Cunderbrace%7B%5Chat%7Bq%7D%28S%2C%20A%2C%20w%29%7D_%7B%5Crm%20%7Bold%7Evalue%7D%7D%29%7D%5E%7B%5Crm%20%7BTD%7Eerror%7D%7D%20%5Cnabla_w%5Chat%7Bq%7D%28S%2C%20A%2C%20w%29)

In practice, with experience replay the local network can be updated every N steps where N is the required number of experience tuples for a single minibatch. As soon as that update finishes the target network copies the parameters from the local network. In this submission a soft update is performed which updates the target network with values interpolated between the two networks' parameters.

### Network Architecture

The neural networks are multi-layer perceptrons with three hidden layers and a fully-connected output layer. The hidden layer outputs are each activated by a Rectified Linear Unit (ReLU). The follow table outlines the size of each layer including the input which is assumed to contain a batch of 64 experience tuples (the default setting in the provided code).


| Layer    | Output Shape | Param # |
| ---------| ------------ | ------- |
| Input    | [64, 37]     | 2,368   |
| Linear-1 | [64, 64]     | 2,432   |
| Linear-2 | [64, 128]    | 8,320   |
| Linear-3 | [64, 64]     | 8,256   |
| Linear-4 | [64, 4]      | 260     |
|          |              | 19,268  |

### Hyperparameter Selection

Much of the code in this repo is taken from excercises in the course material. As a starting point, the hyperparameter values (batch size, discount factor, learning rate, etc.) were copied exactly. These values were found to be sufficient for the problem at had. Using an Adam optimizer, the agent solved the environment in just over 500 episodes with these values.

| Parameter            | Value  |
| -------------------- | ------ |
| Learning Rate        | 0.0005 |
| Interpolation Factor | 0.001  |
| Discount Factor      | 0.99   |
| Batch Size           | 64     |
| Update Frequency     | 4      |
| Buffer Size          | 100000 |

It was observed that increasing the learning rate to 0.001 also increased the number of episodes required to solve the environment to 514. Decreasing the discount facter to 0.75 slowed the process dramatically; after 300 episodes at the lower value, the 100-episode average score was 3.60 compared to 7.43 at the higher value.

## Results

![Trained Banana Collecting Agent](/media/trained_agent.gif 'Oh Banana')

The agent was able to solve the environmet in 505 episodes using the hyperparameters from the preceding table and an Adam optimizer.  
![Training Scores Using Adam Optimizer](/media/training_scores_adam.jpg 'Adam')

An SGD optimizer was also tested (both with and without momentum) but was much slower in the best case. SGD with momentum = 0.9 and learning rate = 0.001 was able to solve the environment in 950 episodes.  
![Training Scores Using SGD w/ Momentum](/media/training_scores_sgd_momentum.jpg 'SGD w/ Momentum = 0.9')

Switching to Nesterov momentum increased the time to solve the environment to 1021 episodes.  
![Training Scores Using SGD w/ Nesterov Momentum](/media/training_scores_sgd_nesterov.jpg 'SGD w/ Nesterov Momentum = 0.9')

Without momentum, SGD did not show any indication of improving network performance; average scores oscillated in the range `[-0.2, 0.9]` for 700 episodes.  
![Training Scores Using SGD](/media/training_scores_sgd.jpg 'SGD')

## Next Steps

Futher experimentation is necessary to ensure that the chosen hyperparamters are indeed optimal. Permutations of different values, training the networks with different optimizers and loss functions, and learning rate decay are all methods which can be incorporated into a automated training pipeline which tests and reports the scores attained from different permutations of values and algorithms. Once these values are set, a few amendments to the learning process may be made to address some known issues with Deep Q-Learning.

There are many extensions to Deep Q-Learning which can be applied to this environment in an effort to improve performance. As a start, prioritized experience replay would enable the agent to learn from samples which provide the greatest information gain. Double Q-Networks can be used to address overestimation by disentagling selection from evaluation. With separate neural networks for each process, the agent compartmentalizes is learning to more effectively choose and evaluate it's actions and obersvations. Similarly, Dueling Q-Networks segment the learning process. However, instead of using two networks, a single network splits into two branches which each specialize in predicting the state values and action advantages, respectively. The results are then combined into the last section of the network which predicts the Q-values.