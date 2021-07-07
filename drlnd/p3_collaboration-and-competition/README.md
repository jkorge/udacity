# Project 3 - Collaberation & Competition

https://user-images.githubusercontent.com/29203930/123530509-1db36a00-d6c9-11eb-9a9e-83fcb28e50d7.mp4

## Setup

This submission for the project is implemented in Python 3.6.13. It is recommened that a conda environment is created before installing other dependencies.

After cloning the repo, run the following to setup the required dependencies:
```bash
    # Create new conda env (recommended)
    conda create -n drlndp3 python=3.6.13
    conda activate drlndp3

    # Install python dependencies (required)
    cd drlnd-p3-multiagent/dependencies
    pip install .

```

This will install all required python modules as well as the unityagents module included in this repo and provided by Udacity as part of the Deep Reinforcement Learning Nanodegree.

The Unity environment required for the project can be downloaded from one of the following links:
- [Linux][1]
- [OSX][2]
- [Windows, 32-bit][3]
- [Windows, 64-bit][4]

Once downloaded, move the file into the project directory, and unzip the file

## Environment

The object of this project is to train two agents to each control a "tennis" racket and keep the ball in the air. Each agent receives a reward of +0.1 every time that agent hits the ball over the net. Each agent receives a reward of -0.01 if the ball falls on their side of the playable area or if they knock the ball into the net or outside the playable area.

The state of each agent at every timestep is a vector in ℝ<sup>24</sup> comprised of measures of that agent's and the ball's position and velocity over the last 3 timesteps.  
Each agent's actions are represented by a vector in ℝ<sup>2</sup> of the velocity of that agent's racket. The values of each action vector are in `[-1,1]`

In each episode the score is determined by the highest score of either agent. The agents are deemed to have solved the environment when they acheive an average score of +0.5 over the last 100 episodes.

## Running the Solution

The entrypoint for this solution is the `main.py` script. There a few options that may be provided after specifying a subcommand. Details are available by running `python main.py --help`.

**IMPORTANT**  
To ensure the solution runs successfully, be sure to specify the path to the environment executable file downloaded during setup:
```bash
    # Windows (x86)
    python main.py --env-file Reacher_Windows_x86_64/Tennis.exe ...

    # Windows (x86_64)
    python main.py --env-file Reacher_Windows_x86_64/Tennis.exe ...

    # Linux (x86)
    python main.py --env-file Reacher_Linux/Tennis.x86 ...

    # Linux (x86_64)
    python main.py --env-file Reacher_Linux/Tennis.x86_64 ...

    # OSX
    python main.py --env-file Tennis.app

```

### Options

Both training and testing have a few configurations in common. To view these options and their defaults run `python main.py --help`.

### Training

To train the agent with default parameteters run `python main.py train`.  
The program will initialize the environment, replay buffer, and target & local neural networks. The agent will then proceed to train for up to 2000 episodes or until it solves the environment, whichever comes first.

To explore other options (including neural network hyperparameters) run `python main.py train --help`.

### Testing

After training has completed the agent can use the saved network weights to run more episodes.  
A checkpoint is already included in the [checkpoints][9] directory and can be loaded to test the agent on a trained network right away. To examine the best performing models run

```
python main.py test -a checkpoints\joint\actor_checkpoint_best.pth -c checkpoints\joint\critic_checkpoint_best.pth
```

To test the agent with default parameters run `python main.py test`.  
The Unity environment will display a spectator's view of the environment as the agents work to keep the ball in the air.




[1]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
[2]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
[3]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
[4]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip
[9]: ./checkpoints