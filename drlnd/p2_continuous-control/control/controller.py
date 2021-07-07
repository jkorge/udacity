import time
import logging
from typing import NamedTuple
from collections import deque

import unityagents

from control import device, default_dtype, torch, np
from control.agent import Agent

unityagents.logger.setLevel(logging.ERROR)

class Hyperparameters(NamedTuple):
    BUFFER_SIZE: int        # replay buffer size
    BATCH_SIZE: int         # minibatch size
    GAMMA: float            # discount factor
    LR: float               # learning rate
    N_STEPS: int            # trajectory length

class Controller:
    '''Control Arms in the Reacher Environment'''

    def __init__(self, env_file, target=30, hparams: Hyperparameters = None, critic_checkpoint=None, actor_checkpoint=None, train=True):
        '''
        Initialize the Controller
        Parameters
        ----------
            env_file : str
                Path to executable for the UnityEnvironment
            target : int
                Minimum score to achieve before the environment is considered solved
            hparams : Hyperparameters
                Hyperparameters object
            critic_checkpoint : str or None
                Path to file containing state_dict of critic model
            actor_checkpoint : str or None
                Path to file containing state_dict of actor model
        '''
        
        # load the environment
        # self.env = UnityEnvironment(file_name=env_file)
        self.env = unityagents.UnityEnvironment(file_name=env_file, no_graphics=train)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.target = target

        # environment details
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])
        self.n_agents = len(env_info.agents)

        # initialize agent
        if hparams is None:
            self.agent = Agent(
                self.state_size, self.action_size, self.n_agents,
                critic_checkpoint=critic_checkpoint, actor_checkpoint=actor_checkpoint
            )
        else:
            self.agent = Agent(
                self.state_size, self.action_size, self.n_agents, *hparams,
                critic_checkpoint=critic_checkpoint, actor_checkpoint=actor_checkpoint
            )

    def _log_prog(self, episode, agent_scores, mean_score, t, solved=False):
        '''
        Print progress to stdout. Writes to new line every 100 episodes
        Parameters
        ----------
            episode : int
                Episode number
            mean_score : float
                Average score achieved over window of recent episodes
            solved : bool
                Flag indicating whether the environment has been solved
        '''
        if solved:
            print(
                f'Environment solved in {episode:>4} episodes',
                f'100-Episode Average Score: {mean_score:.6f}',
                f'Total Time: {t:>.4f}',
                sep = ' - '
            )
        else:
            print(
                f'\rEpisode:{episode:>5}',
                f'Max:{torch.max(agent_scores):>10.6f}',
                f'Min:{torch.min(agent_scores):>10.6f}',
                f'{min(episode, 100):>3}-Episode Average Score:{mean_score:>10.6f}',
                f'Time (s):{t:>8.4f}',
                sep=' - ', end=''
            )

    def run(self, n_episodes, train):
        '''
        Train Agent
        
        Parameters
        ----------
            n_episodes : int
                Maximum number of training episodes
        '''

        if n_episodes < 0:
            n_episodes = float('inf')

        if train:
            stop_condition = lambda x, s: (x >= 100) and (s >= self.target)
        else:
            stop_condition = lambda x, s: (x == float('inf')) and (s >= self.target)

        # list of scores for each episode (average of all agents' scores for the episode)
        episode_scores = list()
        agent_scores = list()
        # deque of episode_scores
        scores_window = deque(maxlen=100)
        # average of scores in scores_window
        mean_scores = list()

        episode = 1
        START = time.time()
        while episode < (n_episodes + 1):

            # initialize episode
            env_info = self.env.reset(train_mode=train)[self.brain_name]
            states = torch.tensor(env_info.vector_observations)
            _agent_scores = torch.zeros(len(states))

            start = time.time()

            while True:

                # predict/take actions
                actions = self.agent.act(states, explore=train)
                env_info = self.env.step(actions.cpu().numpy())[self.brain_name]

                # observe new state/reward
                next_states = torch.tensor(env_info.vector_observations)
                rewards = torch.tensor(env_info.rewards)
                dones = torch.tensor(env_info.local_done, dtype=torch.bool)

                # add a SMALL negative reward for not staying in target zone
                rewards[rewards == 0] = -1e-5

                # update agent
                if train and self.agent.step(states, actions, rewards, next_states, dones):
                    for _ in range(4):
                        self.agent.learn()
                    self.agent.update()

                # Accumulate scores and move on
                _agent_scores += rewards
                states = next_states
                agent_scores.append(_agent_scores)

                if torch.any(dones):
                    break

            stop = time.time()

            # update score trackers
            episode_score = torch.max(_agent_scores).item()
            episode_scores.append(episode_score)
            scores_window.append(episode_score)
            mean_score = np.mean(scores_window)
            mean_scores.append(mean_score)

            # log progress
            self._log_prog(episode, _agent_scores, mean_score, stop-start)

            if stop_condition(episode, mean_score):
                STOP = time.time()
                self._log_prog(episode, None, mean_score, STOP-START, solved=True)
                # self.agent.save()
                break
            else:
                episode += 1

        self.env.close()
        agent_scores = torch.stack(agent_scores).cpu().numpy().T
        return agent_scores, episode_scores, mean_scores