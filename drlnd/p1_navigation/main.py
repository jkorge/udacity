import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from navigation.navigator import Navigator, Hyperparameters

def click_option(*args, **kwargs):
    if not 'show_default' in kwargs:
        kwargs.update({'show_default': True})
    return click.option(*args, **kwargs)

@click.group()
@click_option('-e','--env-file', type=click.Path(exists=True), default='Banana_Windows_x86_64/Banana.exe', help='Path to executable containing UnityEnvironment')
@click_option('-c', '--checkpoint', type=click.Path(exists=False), default='', help='Path to Neural Netowrk checkpoint file')
@click_option('-n', '--n-episodes', default=2000, help='Number of episodes to run. Provides an upper limit for training. Pass negative during testing to run until environment is solved')
@click_option('-t', '--max-t', default=1000, help='Maximum number of steps to take in each episode')
@click_option('-T', '--target', default=13., help='Minimum score (averaged over 100 episodes) needed for environment to be considered solved')
@click.pass_context
def cli(ctx, env_file, checkpoint, n_episodes, max_t, target):
    ctx.obj['ENV'] = env_file
    ctx.obj['CHKPT'] = checkpoint
    ctx.obj['EPISODES'] = n_episodes
    ctx.obj['MAX_T'] = max_t
    ctx.obj['TARGET'] = target

@cli.command()
@click_option('-r', '--replay-buffer-size', default=int(1e5), help='Size of experience replay buffer')
@click_option('-b', '--batch-size', default=64, help='Minibatch size used during model training')
@click_option('-u', '--update-freq', default=4, help='Frequency with which to update Neural Network parameters')
@click_option('-g', '--gamma', default=0.99, help='RL discount factor')
@click_option('-t', '--tau', default=1e-3, help='Deep Q-Learning interpolation factor')
@click_option('-l', '--learning-rate', default=5e-4, help='Neural Network leraning rate')
@click_option('-e', '--eps-start', default=1.0, help='Starting value for epsilon')
@click_option('-f', '--eps-end', default=0.01, help='Final value for epislon')
@click_option('-d', '--eps-decay', default=0.995, help='Multiplicative decay rate for epislon')
@click_option('-p', '--plot-scores', is_flag=True, help='Plots the scores the agent achieved during training')
@click.pass_context
def train(ctx, replay_buffer_size, batch_size, update_freq, gamma, tau, learning_rate, eps_start, eps_end, eps_decay, plot_scores):
    # initialize Navigator
    hparams = Hyperparameters(replay_buffer_size, batch_size, update_freq, gamma, tau, learning_rate)
    nav = Navigator(ctx.obj['ENV'], ctx.obj['TARGET'], hparams)
    if len(ctx.obj['CHKPT']):
        nav.load(ctx.obj['CHKPT'])

    # train agent
    n_episodes = ctx.obj['EPISODES']
    max_t = ctx.obj['MAX_T']
    scores, mean_scores = nav.train(n_episodes, max_t, eps_start, eps_end, eps_decay)

    if plot_scores:
        fig, ax = plt.subplots(1, figsize=(20,10))
        ax.set_title('Training Progress')
        ax.set_ylabel('Score')
        ax.set_xlabel('Episode')
        domain = np.arange(len(scores))
        spline = InterpolatedUnivariateSpline(domain[::20], scores[::20])
        domain_sp = np.linspace(0, len(scores), 1000)
        ax.plot(domain, scores, 'g-', domain_sp, spline(domain_sp), 'r--', domain, mean_scores, 'b-')
        ax.legend(['Raw Scores', 'Interpolated Univariate Spline', '100 Episode Average'])
        plt.savefig('media/training_scores_sgd.jpg', dpi=300, bbox_inches='tight')
        plt.show()


@cli.command()
@click.pass_context
def test(ctx):
    #initialize Navigator
    nav = Navigator(ctx.obj['ENV'], ctx.obj['TARGET'])
    if len(ctx.obj['CHKPT']):
        nav.load(ctx.obj['CHKPT'])

    # test agent
    n_episodes = ctx.obj['EPISODES']
    max_t = ctx.obj['MAX_T']
    scores = nav.test(n_episodes, max_t)

if __name__ == '__main__':
    cli(obj=dict())