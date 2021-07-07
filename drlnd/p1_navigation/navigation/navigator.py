from typing import NamedTuple
from collections import deque

import torch
import numpy as np
from unityagents import UnityEnvironment

from navigation.agent import Agent

class Hyperparameters(NamedTuple):
    BUFFER_SIZE: int        # replay buffer size
    BATCH_SIZE: int         # minibatch size
    UPDATE_FREQ: int        # target network update frequency
    GAMMA: float            # discount factor
    TAU: float              # interpolation factor
    LR: float               # learning rate

class Navigator:
    '''Navigate Banana Collection Environment'''

    def __init__(self, env_file, target=13, hparams: Hyperparameters = None):
        '''
        Initialize the Navigator
        Params
        ======
            env_file (str): path to executable for the UnityEnvironment
            target (int): minimum score to achieve before the environment is considered solved
            hparams (Hyperparameters): training parameters
        '''
        
        # load the environment
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.target = target

        # environment details
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])

        # initialize agent
        if hparams is None:
            self.agent = Agent(self.state_size, self.action_size)
        else:
            self.agent = Agent(self.state_size, self.action_size, *hparams)

    def _log_prog(self, episode, mean_score, solved=False):
        '''
        Print progress to stdout. Writes to new line every 100 episodes
        Params
        =======
            episode (int): episode number
            mean_score (float): average score achieved over window of recent episodes
            solved (bool): flag indicating whether the environment has been solved
        '''
        if solved:
            print(f'Environment Solved in {episode:>4} episodes - Average Score: {mean_score:.2f}')
        else:
            endl = '' if ((episode>0) and (episode%100)) else '\n'
            print(f'\rEpisode {episode:>4} - Average Score: {mean_score:.2f}', end=endl)

    def train(self, n_episodes, max_t, eps_start, eps_end, eps_decay):
        '''
        Train Agent with Deep Q-Learning
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        '''
        scores = list()
        mean_scores = list()
        scores_window = deque(maxlen=100)
        eps = eps_start
        for episode in range(1, n_episodes+1):
            # initialize episode
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for t in range(max_t):
                # predict/take action
                action = int(self.agent.act(state, eps))
                env_info = self.env.step(action)[self.brain_name]

                # observe new state/reward
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                # update agent
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            # update score trackers and epislon
            scores_window.append(score)
            scores.append(score)
            mean_score = np.mean(scores_window)
            mean_scores.append(mean_score)
            eps = max(eps_end, eps_decay*eps)

            # log progress
            self._log_prog(episode, mean_score)
            if mean_score >= self.target:
                self._log_prog(episode, mean_score, solved=True)
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoints/checkpoint.pth')
                break

        self.env.close()
        return scores, mean_scores

    def test(self, n_episodes, max_t):
        '''
        Test Agent

        Params
        ======
            n_episodes (int): number of episodes to run; use negative value to run until environment is solved
            max_t (int): maximum number of steps to take in each episdoe
        '''

        if n_episodes < 0:
            n_episodes = float('inf')

        episode = 0
        scores = list()
        scores_window = deque(maxlen=100)

        while episode < n_episodes:
            #initialize episode
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for t in range(max_t):
                # Take action & update state
                action = int(self.agent.act(state))
                env_info = self.env.step(action)[self.brain_name]
                state = env_info.vector_observations[0]

                # Accumulate reward
                score += env_info.rewards[0]
                done = env_info.local_done[0]
                if done:
                    break

            # update trackers
            episode += 1
            scores.append(score)
            scores_window.append(score)
            mean_score = np.mean(scores_window)

            # log progress
            self._log_prog(episode, mean_score)
            if (n_episodes == float('inf')) and (mean_score >= self.target):
                # quit infinite loop when env is solved
                self._log_prog(episode, mean_score, solved=True)
                break

        self.env.close()
        return scores


    def load(self, checkpoint):
        '''
        Load model weights from checkpoint file. Calls `self.agent.load(checkpoint)`

        Params
        ======
            checkpoint (str): path to file containing model's state_dict
        '''
        self.agent.load(checkpoint)