import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from control.controller import Controller, Hyperparameters

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
    default='envs/Reacher20_Windows_x86_64/Reacher.exe',
    help='Path to executable containing UnityEnvironment'
)
@click_option('-n', '--n-episodes',
    default=2000,
    help='Number of episodes to run. Provides an upper limit for training. Pass negative during testing to run until environment is solved'
)
@click_option('-T', '--target',
    default=30.,
    help='Minimum score (averaged over 100 episodes) needed for environment to be considered solved'
)
@click.option('-c', '--critic-checkpoint',
    type=click.Path(exists=False),
    default='',
    help='Path to Neural Netowrk checkpoint file for Critic model'
)
@click.option('-a', '--actor-checkpoint',
    type=click.Path(exists=False),
    default='',
    help='Path to Neural Netowrk checkpoint file for Actor model'
)
@click_option('-r', '--replay-buffer-size',
    default=int(1e6),
    help='Size of experience replay buffer'
)
@click_option('-b', '--batch-size',
    default=256,
    help='Minibatch size used during model training'
)
@click_option('-s', '--trajectory-length',
    default=5,
    help='Number of steps to include in each trajectory'
)
@click_option('-g', '--gamma',
    default=0.99,
    help='RL discount factor'
)
@click_option('-l', '--learning-rate',
    default=1e-4,
    help='Initial learning rate for Actor-Critic Neural Networks'
)
@click_option('-p', '--plot-scores',
    is_flag=True,
    help='Plots the scores the agent achieved during training'
)
def cli(op,
    env_file, n_episodes, target,
    critic_checkpoint, actor_checkpoint,
    replay_buffer_size, batch_size, trajectory_length,
    gamma, learning_rate,
    plot_scores
):
    
    train = op == 'train'
    params = dict(replay_buffer_size=replay_buffer_size, batch_size=batch_size, trajectory_length=trajectory_length, gamma=gamma, learning_rate=learning_rate)

    if train:
        print(f'Training for {n_episodes} episodes or until agents achieve 100-episode average of {target}')
        for k,v in params.items():
            print(f'{k:<20}:{v:>10}')
    else:
        til = f'for {n_episodes} episodes' if n_episodes > 0 else f'until agents achieve 100-episode average of {target}'
        print(f'Testing ' + til)

    print(f'{"plot_scores":<20}:{plot_scores:>10}', end='\n\n')

    # initialize Controller
    con = Controller(
        env_file=env_file,
        target=target,
        hparams=Hyperparameters(replay_buffer_size, batch_size, gamma, learning_rate, trajectory_length),
        critic_checkpoint=critic_checkpoint,
        actor_checkpoint=actor_checkpoint,
        train=train
    )

    # train agent
    agent_scores, episode_scores, mean_scores = con.run(n_episodes, train=train)
    if plot_scores:
        plot(agent_scores, episode_scores, mean_scores, train)

if __name__ == '__main__':
    cli()
    