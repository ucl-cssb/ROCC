## ROCC: Reinforcement learning for the Optimisation of Co-Cultures
An application to train reinforcement learning to control a chemostat containing multiple interacting populations of bacteria.


### Installation
To use the package within python scropts, `ROCC` must be in PYTHONPATH.

### Dependencies
Standard python dependencies are required: `numpy`, `scipy`, `matplotlib`.`yaml` is required to parse parameter files. `argparse` is required for the command line application. `pytest` is required to run the unit tests, `TensorFlow` is required). Instructions for installing 'TensorFlow' can be found here:
 https://www.tensorflow.org/install/

### User Instructions
Code files can be imported into scripts, see examples

To run examples found in fitted_Q_iteration/examples from the command line:
```console
$ python double_aux_example.py -s <save_path> -r <repeat_number>
```
  - -s, --save_path: path to save results
  - -r, --repeats: the repeat number, intended for use when running in parrellel as an array job on a cluster, set to 1 to run example


The examples will automatically save some results in a directory structure in save_path:

```
save_path
 ├── repeat1
 │   ├── train_trajectory.npy
 │   ├── train_populations.npy
 │   ├── train_returns.png
 ├── repeat2
 .
 .
 .
```

Where train_trajectory.npy is the system trajectory under control of the trained agent, train_populations.png is a graph of the populations, train_returns.png is a graph of the return recieved in each episode.

The files ftted_Q_on_double_aux.py, PI_comp.py, parrellel_fitted_Q.py and optimising_product.py are the files used to get the results in the paper. These are tun with the same options as the examples and will output more results files.

The user can define a system a .yaml parameter file. Examples of these for multiple two and three species systems are found in the chemostat_env/parameter_files directory. Training parameters are set in the top of the scripts.

The fitted_Q_agents.py file can be imported and used on any RL task.


### run_files
Contains some bash scripts for running repeats

### testing
Contains code to test environments and agents and the investigations into the reinforcement learning parameter tuning


### chemostat_env
Contains the environments used for RL on the bioreactor system. Can be imprted and used with any control algorithm
