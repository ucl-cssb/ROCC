import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)



from chemostat_envs import *
from fitted_Q_agents import *
from argparse import ArgumentParser

from reward_func import *


def run_test(save_path):
    param_path = os.path.join(C_DIR, 'parameter_files/double_aux.yaml')
    one_min = 0.016666666667
    n_mins = 5

    sampling_time = n_mins*one_min

    tmax = int((24*60)/n_mins) # set this to 24 hours
    #tmax = 10
    print('tmax: ', tmax)
    n_episodes = 30
    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []
    pop_scaling = 100000
    env = ChemostatEnv(param_path, reward_func, sampling_time,  pop_scaling)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])

    train_trajectorys = []
    for i in range(n_episodes):
        print('EPISODE: ', i)
        # training EPISODE
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        print('explore rate: ', explore_rate)
        env.reset()
        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_trajectorys.append(train_trajectory)
        train_times.append(len(train_trajectory))
        train_rewards.append(train_r)
        print('return: ', train_r)


    os.makedirs(save_path, exist_ok = True)

    train_rewards = np.array(train_rewards)

    train_times = np.array(train_times)


    # plot the last train trajectory
    plt.figure()
    plt.ylabel('Populations')
    plt.xlabel('Timestep')
    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.savefig(save_path + '/train_populations.png')
    np.save(save_path + '/train_trajectory.npy', train_trajectory)

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Populations')
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_returns.png')



if __name__ == '__main__':
    run_test('./double_aux_example_results')
