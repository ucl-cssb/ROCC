import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import ROCC

from argparse import ArgumentParser

def run_test(save_path):
    P_DIR = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ROCC'), 'chemostat_env'), 'parameter_files')

    param_path = os.path.join(P_DIR, 'product.yaml')
    one_min = 0.016666666667
    n_mins = 10

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
    env = ProductEnv(param_path, sampling_time, pop_scaling)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])

    train_trajectorys = []

    for i in range(n_episodes):
        print('EPISODE: ', i)
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        print('\texplore_rate: ', explore_rate)
        initial_S = np.append(np.append(np.append(np.random.uniform(10000, 50000, 2), env.initial_C), env.initial_C0),env.initial_chems)
        env.reset(initial_S)

        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_trajectorys.append(train_trajectory)
        train_times.append(len(train_trajectory))
        train_rewards.append(train_r)
        train_actions = np.array(agent.actions)
        print('\treturn: ', train_r)

        values = np.array(agent.values)
        env.plot_trajectory([0,1])


    os.makedirs(save_path, exist_ok = True)

    np.save(save_path + '/train_trajectories.npy', np.array(train_trajectorys))

    train_rewards = np.array(train_rewards)

    train_times = np.array(train_times)


    # plot the last train trajectory
    plt.figure()

    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Population ($10^6 cells L^{-1}$)')
    plt.ylim(bottom = 15000)
    plt.xlim(left = 0)
    plt.xlim(right = 1440)
    plt.hlines([20000, 30000], 0, 288*5, color = 'g', label = 'target')
    plt.savefig(save_path + '/train_populations.png')
    np.save(save_path + '/train_trajectory.npy', train_trajectory)


    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')





if __name__ == '__main__':
    run_test('./optimise_product_example_results')
