import pdb
import time
import logging
from typing import NamedTuple
from collections import deque

import unityagents

from multiagent import device, torch, np
from multiagent.multiagent import MultiAgent
from multiagent.jointagent import JointAgent

unityagents.logger.setLevel(logging.ERROR)

class Hyperparameters(NamedTuple):
    BUFFER_SIZE: int        # replay buffer size
    BATCH_SIZE: int         # minibatch size
    GAMMA: float            # discount factor
    LR: float               # learning rate
    TAU: float              # interpolation factor
    N_STEPS: int            # trajectory length

class Driver:
    '''Manage training/testing of agents in UnityEnvironment'''

    def __init__(self,
        env_file, hparams,
        multi=False, shared_buffer=True,
        target=0.5, train=True, soft_update=False,
        critic_checkpoint=None, actor_checkpoint=None,
    ):
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
            multi : bool
                Use MultiAgent; else, use JointAgent
            shared_buffer : bool
                Agents contribute to a single, shared buffer. Ignored unless `multi` is True
            target : float
                100-Episode average score. Agents will cease training after achieving this
            train : bool
                Flag for running in training mode
            soft_update: bool
                Flag for using a soft update when update target network params
            critic_checkpoint : str or None
                Path to file containing state_dict of critic model
            actor_checkpoint : str or None
                Path to file containing state_dict of actor model
        '''
        
        # load the environment
        self.env = unityagents.UnityEnvironment(file_name=env_file, no_graphics=train)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.target = target
        self.soft_update = soft_update

        # environment details
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])
        self.n_agents = len(env_info.agents)


        # initialize agent
        self.multi = multi
        agent_cls = MultiAgent if self.multi else JointAgent
        if hparams is None:
            self.agent = agent_cls(
                state_size=self.state_size,
                action_size=self.action_size,
                n_agents=self.n_agents,
                critic_checkpoint=critic_checkpoint,
                actor_checkpoint=actor_checkpoint,
                shared_buffer=shared_buffer
            )
        else:
            self.agent = agent_cls(
                state_size=self.state_size,
                action_size=self.action_size,
                n_agents=self.n_agents,
                buffer_size=hparams.BUFFER_SIZE,
                batch_size=hparams.BATCH_SIZE,
                gamma=hparams.GAMMA,
                lr=hparams.LR,
                tau=hparams.TAU,
                trajectory_length=hparams.N_STEPS,
                critic_checkpoint=critic_checkpoint,
                actor_checkpoint=actor_checkpoint,
                shared_buffer=shared_buffer
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
                f'\nEnvironment solved in {episode:>4} episodes',
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
                sep=' - ', end='' if episode%100 else '\n'
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

        # scores for each episode
        episode_scores = list()

        # scores for each agent for each episode
        agent_scores = list()

        # deque of episode_scores
        scores_window = deque(maxlen=100)

        # average of scores in scores_window
        mean_scores = list()

        best_score = 0.0

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
                actions = self.agent.act(states, train=train)
                env_info = self.env.step(actions.numpy())[self.brain_name]

                # observe new state/reward
                next_states = torch.tensor(env_info.vector_observations)
                rewards = torch.tensor(env_info.rewards)
                dones = torch.tensor(env_info.local_done, dtype=torch.bool)

                # update agent
                if train and self.agent.step(states, actions, rewards, next_states, dones):
                    for _ in range(4):
                        self.agent.learn()
                    self.agent.update(soft_update=self.soft_update)

                # Accumulate scores and move on
                _agent_scores += rewards
                states = next_states
                if torch.any(dones):
                    agent_scores.append(_agent_scores)
                    break

            stop = time.time()

            # update score trackers
            episode_score = torch.max(_agent_scores).item()
            episode_scores.append(episode_score)
            scores_window.append(episode_score)
            mean_score = np.mean(scores_window)
            mean_scores.append(mean_score)

            # report progress
            self._log_prog(episode, _agent_scores, mean_score, stop-start)

            if train and (episode > 100) and (episode_score > best_score):
                best_score = episode_score
                self.save(score=best_score)


            if stop_condition(episode if train else n_episodes, mean_score):
                # Solved - print results and save models
                STOP = time.time()
                self._log_prog(episode, None, mean_score, STOP-START, solved=True)
                if train:
                    self.save()
                break
            else:
                episode += 1

        self.env.close()
        agent_scores = torch.stack(agent_scores).cpu().numpy().T
        return agent_scores, episode_scores, mean_scores

    def save(self, score=None):
        if self.multi:
            self.agent.save(critic_checkpoint_dir='checkpoints/multi/critic', actor_checkpoint_dir='checkpoints/multi/actor', score=score)
        else:
            c = 'checkpoints/joint/critic_checkpoint.pth' if score is None else f'checkpoints/joint/critic_checkpoint_{score:.6f}.pth'
            a = 'checkpoints/joint/actor_checkpoint.pth' if score is None else f'checkpoints/joint/actor_checkpoint_{score:.6f}.pth'
            self.agent.save(critic_checkpoint=c, actor_checkpoint=a)
