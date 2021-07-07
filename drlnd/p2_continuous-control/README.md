# Project 2 - Control

## Setup

This submission for the project is implemented in Python 3.6.13. It is recommened that a conda environment is created before installing other dependencies.

After cloning the repo, run the following to setup the required dependencies:
```bash
    # Create new conda env (recommended)
    conda create -n control python=3.6.13
    conda activate control

    # Install python dependencies (required)
    cd drlnd-p2-control/dependencies
    pip install .

```

This will install all required python modules as well as the unityagents module included in this repo and provided by Udacity as part of the Deep Reinforcement Learning Nanodegree.

There are two versions of the environment for this project; either one can be used with this code. Note that only version 2 was tested rigorously. The Unity environment required for the project can be downloaded from one of the following links:
- [Linux (Version 1)][1]
- [Linux (Version 2)][2]
- [OSX (Version 1)][3]
- [OSX (Version 2)][4]
- [Windows, 32-bit (Version 1)][5]
- [Windows, 32-bit (Version 2)][6]
- [Windows, 64-bit (Version 1)][7]
- [Windows, 64-bit (Version 2)][8]

Once downloaded, move the file into the project directory, and unzip the file

## Environment

The object of this project is to train an agent to control a double-jointed arm such that the end of the arm is continuously held within a moving target region. The agent receives a reward of +0.1 for every timestep during which the arm is held within the target region.

The state of the agent at every timestep is a vector in ℝ<sup>33</sup> comprised of measures of the agent's position, rotation, velocity, and angular velocity.  
The agent's actions are represented by a vector in ℝ<sup>4</sup> of the torques applied to each joint. The values of each action vector are in `[-1,1]`

The agent is deemed to have solved the environment when it achieves an average score of +30 over the last 100 episodes.  
The second version of the environment features the same problem space, however, it instantiates 20 agents and 20 corresponding target regions. The observations and actions are the same with the only caveat that there are now 20 observations made and actions taken at each timestep.

## Running the Solution

The entrypoint for this solution is the `main.py` script. There a few options that may be provided before specifying a subcommand. Details are available by running `python main.py --help`.

**IMPORTANT**  
To ensure the solution runs successfully, be sure to specify the path to the environment executable file downloaded during setup:
```bash
    # Windows (x86)
    python main.py --env-file Reacher_Windows_x86_64/Reacher.exe ...

    # Windows (x86_64)
    python main.py --env-file Reacher_Windows_x86_64/Reacher.exe ...

    # Linux (x86)
    python main.py --env-file Reacher_Linux/Reacher.x86 ...

    # Linux (x86_64)
    python main.py --env-file Reacher_Linux/Reacher.x86_64 ...

    # OSX
    python main.py --env-file Reacher.app

```

### Options

Both training and testing have a few configurations in common. To view these options and their defaults run `python main.py --help`.

### Training

To train the agent with default parameteters run `python main.py train`.  
The program will initialize the environment, replay buffer, and target & local neural networks. The agent will then proceed to train for up to 2000 episodes or until it solves the environment, whichever comes first.

To explore other options (including neural network hyperparameters) run `python main.py train --help`.

### Testing

After training has completed the agent can use the saved network weights to run more episodes.  
A checkpoint is already included in the [checkpoints][9] directory and can be loaded to test the agent on a trained network right away.

To test the agent with default parameters run `python main.py test`.  
The Unity environment will display the agent's view of the environment as it works to keep the arm within the target area (hopefully).




[1]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
[2]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
[3]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
[4]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
[5]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip
[6]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip
[7]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip
[8]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip
[9]: ./checkpoints