import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from multiagent.driver import Driver, Hyperparameters

def click_option(*args, **kwargs):
    if not 'show_default' in kwargs:
        kwargs.update({'show_default': True})
    return click.option(*args, **kwargs)

def plot(agent_scores, episode_scores, mean_scores, train):
    # new figure
    fig, ax = plt.subplots(1, figsize=(20,10))
    ax.set_title('Training Progress', fontsize=14)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode')

    # x-axis values
    N = len(episode_scores)
    domain = np.arange(N)
    domain_sp = np.linspace(0, N, 1000)

    # spline of scores
    spline = InterpolatedUnivariateSpline(domain[::20], episode_scores[::20])(domain_sp)

    # plot scores, 100-episode averages, and spline of scores
    ax.plot(domain, episode_scores, 'g-', domain_sp, spline, 'r--', domain, mean_scores, 'b-')
    ax.legend(['Raw Scores', 'Interpolated Univariate Spline', '100 Episode Average'])

    # save image and display
    x = 'train' if train else 'test'
    plt.savefig(f'media/{x}ing_scores.jpg', dpi=300, bbox_inches='tight')
    plt.show()

@click.command(context_settings=dict(max_content_width=200))
@click.argument('op',
    type=click.Choice(['train', 'test'])
)
@click_option('-e','--env-file',
    type=click.Path(exists=True),
    default='./envs/Tennis_Windows_x86_64/Tennis.exe',
    help='Path to executable containing UnityEnvironment'
)
@click_option('-n', '--n-episodes',
    default=2000,
    type=int,
    help='Number of episodes to run. Provides an upper limit for training. Pass negative during testing to run until environment is solved'
)
@click_option('--multi/--no-multi',
    default=False,
    help='Use Multi-agent training instead of joint-agent'
)
@click_option('-T', '--target',
    default=0.5,
    type=float,
    help='Minimum score (averaged over 100 episodes) needed for environment to be considered solved'
)
@click.option('-c', '--critic-checkpoint',
    type=click.Path(exists=False),
    default='',
    help='Path to Neural Netowrk checkpoint file (joint) or directory (multi) for Critic model'
)
@click.option('-a', '--actor-checkpoint',
    type=click.Path(exists=False),
    default='',
    help='Path to Neural Netowrk checkpoint file (joint) or directory (multi) for Actor model'
)
@click_option('--shared-buffer/--no-shared-buffer',
    default=True,
    help='Collect agents\' experiences in a single, shared buffer. Has no effect when training a joint agent'
)
@click_option('-r', '--replay-buffer-size',
    default=1e5,
    type=int,
    help='Size of experience replay buffer'
)
@click_option('-b', '--batch-size',
    default=128,
    type=int,
    help='Minibatch size used during model training'
)
@click_option('-s', '--trajectory-length',
    default=5,
    type=int,
    help='Number of steps to include in each trajectory'
)
@click_option('-g', '--gamma',
    default=0.99,
    type=float,
    help='RL discount factor'
)
@click_option('-l', '--learning-rate',
    default=1e-4,
    type=float,
    help='Initial learning rate for Actor-Critic Neural Networks'
)
@click_option('-t', '--tau',
    default=1e-3,
    type=float,
    help='Interpolation factor for soft updates of Actor-Critic networks'
)
@click_option('-h', '--hard-update',
    is_flag=True,
    help='Use hard updates for Actor-Critic networks. Supercedes -t'
)
@click_option('--plot-scores/--no-plot-scores',
    default=True,
    help='Plots the scores the agent achieved during training'
)
def cli(op,
    env_file, n_episodes, multi, target,
    critic_checkpoint, actor_checkpoint,
    shared_buffer, replay_buffer_size, batch_size,
    trajectory_length,
    gamma, learning_rate, tau, hard_update,
    plot_scores
):
    
    # Parameter munging
    train = op == 'train'
    actor_checkpoint = None if not actor_checkpoint else actor_checkpoint
    critic_checkpoint = None if not critic_checkpoint else critic_checkpoint

    # Parameter dictionary
    params = dict(
        multi_agent=multi,
        shared_buffer=shared_buffer,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        trajectory_length=trajectory_length,
        gamma=gamma, 
        learning_rate=learning_rate,
        plot_scores=plot_scores
    )
    if hard_update:
        params.update(dict(hard_update=bool(hard_update)))
    else:
        params.update(dict(tau=tau))

    # Print parameters to stdout
    print(f'\n\nRunning in {op} mode', end=' ')
    if n_episodes > 0:
        print(f'for {n_episodes} episodes or', end=' ')
    print(f'until agents achieve 100-episode average score of {target}\n')

    for k,v in params.items():
        if (k == 'multi_agent'):
            k = 'agent_type'
            v = 'multi' if v else 'joint'
        if (k == 'shared_buffer') and (not params['multi_agent']):
            continue
        print(f'\t{k:<20}:{repr(v):>10}')
    print()

    # Initialize multiagent environment
    cc = Driver(
        env_file=env_file,
        hparams=Hyperparameters(replay_buffer_size, batch_size, gamma, learning_rate, tau, trajectory_length),
        target=target,
        critic_checkpoint=critic_checkpoint,
        actor_checkpoint=actor_checkpoint,
        train=train,
        soft_update=not hard_update,
        multi=multi,
        shared_buffer=shared_buffer
    )

    # Run training/testing
    agent_scores, episode_scores, mean_scores = cc.run(n_episodes, train=train)
    if plot_scores:
        plot(agent_scores, episode_scores, mean_scores, train)

if __name__ == '__main__':
    cli()
    