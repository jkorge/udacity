# Project 1 - Navigation

## Setup

This submission for the project is implemented in Python 3.6.13. It is recommened that a conda environment is created before installing other dependencies.

After cloning the repo, run the following to setup the required dependencies:
```bash
    # Create new conda env (recommended)
    conda create -n navigation python=3.6.13
    conda activate navigation

    # Install python dependencies (required)
    cd drlnd-p1-navigation/dependencies
    pip install .

```

This will install all required python modules as well as the unityagents module included in this repo and provided by Udacity as part of the Deep Reinforcement Learning Nanodegree.

The Unity environment required for the project can be downloaded from one of the following links:
- [Linux][1]
- [OSX][2]
- [Windows (32-bit)][3]
- [Windows (64-bit)][4]

Once downloaded, move the file into the project directory, and unzip the file

## Environment

The object of this project is to train an agent to navigate a 3D environment filled with yellow and blue bananas worth +1 and -1 points, respectively.

The state of the agent at every timestep is a 37-vector comprised of the agent's current velocity as well as a "ray-based perception of objects around the agent's forward direction."  
There are four actions available to the agent:
- Move Forward
- Move Backward
- Move Left
- Move Right

The agent moves around on a square, bounded plane. If the agent attempts to move into the boundary, it will remain in place.  
The agent is deemed to have solved the environment when it achieves an average score of +13 over the last 100 episodes.

## Running the Solution

The entrypoint for this solution is the `main.py` script. There a few options that may be provided before specifying a subcommand. Details are available by running `python main.py --help`.

**IMPORTANT**  
To ensure the solution runs successfully, be sure to specify the path to the environment executable file downloaded during setup:
```bash
    # Windows (x86)
    python main.py --env-file Banana_Windows_x86/Banana.exe ...

    # Windows (x86_64)
    python main.py --env-file Banana_Windows_x86_64/Banana.exe ...

    # Linux (x86)
    python main.py --env-file Banana_Linux/Banana.x86 ...

    # Linux (x86_64)
    python main.py --env-file Banana_Linux/Banana.x86_64 ...

    # OSX
    python main.py --env-file Banana.app

```

### Options

Both training and testing have a few configurations in common. To view these options and their defaults run `python main.py --help`.

### Training

To train the agent with default parameteters run `python main.py train`.  
The program will initialize the environment, replay buffer, and target & local neural networks. The agent will then proceed to train for up to 2000 episodes or until it solves the environment, whichever comes first.

To explore other options (including neural network hyperparameters) run `python main.py train --help`.

### Testing

After training has completed the agent can use the saved network weights to run more episodes.  
A checkpoint is already included in the [checkpoints][5] directory and can be loaded to test the agent on a trained network right away.

To test the agent with default parameters run `python main.py test`.  
The Unity environment will display the agent's view of the environment as it works to collect yellow bananas and avoid blue bananas (hopefully).




[1]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
[2]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
[3]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
[4]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
[5]: ./checkpoints