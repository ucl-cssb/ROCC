## ROCC: Reinforcement learning for the Optimisation of Co-Cultures
An application to train reinforcement learning to control a chemostat containing multiple interacting populations of bacteria.


### Installation
To use the package within python scropts, `ROCC` must be in PYTHONPATH.

To add to PYTHONPATH on a bash system add the following to the ~/.bashrc file

```console
export PYTHONPATH="${PYTHONPATH}:<path to ROCC_master>"
```

### Dependencies
Standard python dependencies are required: `numpy`, `scipy`, `matplotlib`.`yaml` is required to parse parameter files. `argparse` is required for the command line application. `TensorFlow` is required). Instructions for installing 'TensorFlow' can be found here:
 https://www.tensorflow.org/install/

### User Instructions
Code files can be imported into scripts, ensure the ROCC directory is in PYTHONPATH and simply import ROCC. See examples.

To run examples found in ROCC_master/examples from the command line, e.g.:

```console
$ python double_aux_example.py 
```


The examples will automatically save some results in the directory.


Where train_trajectory.npy is the system trajectory under control of the trained agent, train_populations.png is a graph of the populations, train_returns.png is a graph of the return recieved in each episode.

Any .yaml parameter file can be defined. Examples of these for multiple two and three species systems are found in the chemostat_env/parameter_files directory. Training parameters are set in the top of the scripts.

The main classes are the fitted_Q_agents and chemostat_env, see examples for how to use these:

### fitted_Q_iteration
The fitted_Q_agents.py file can be imported and used on any RL task.


### chemostat_env
Contains the environments used for RL on the bioreactor system. Can be imported and used with any control algorithm
