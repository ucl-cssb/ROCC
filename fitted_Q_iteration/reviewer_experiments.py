import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)

print(C_DIR)
from chemostat_envs import *
from fitted_Q_agents import *
from argparse import ArgumentParser

from reward_func import *

def entry():
    '''
    Entry point for command line application handle the parsing of arguments and runs the relevant agent
    '''
    # define arguments
    parser = ArgumentParser(description = 'Bacterial control app')
    parser.add_argument('-s', '--save_path')
    parser.add_argument('-r', '--repeat')
    arguments = parser.parse_args()

    # get number of repeats, if not supplied set to 1
    repeat = int(arguments.repeat) - 1
    experiment_number = repeat//6 # six repeats per parameter set
    #experiment_number = repeat//3
    suffix = experiment_number - 1 if experiment_number > 0 else ''

    param_path = os.path.join(os.path.join(os.path.join(C_DIR,'parameter_files'),  'reviewer_exp'), 'double_aux') + str(suffix)  + '.yaml'
    print('PARAMS: ', param_path)

    save_path = os.path.join(arguments.save_path, 'repeat' + str(repeat))
    print(save_path)
    print(experiment_number)
    reward_f = reward_func if repeat%6 in [0, 1, 2] else flipped_reward_func
    #reward_f = flipped_reward_func
    print(reward_f)
    # choose param_path and save_path based on repeat number
    run_test(param_path, save_path, reward_f)

def run_test(param_path, save_path, reward_func):

    update_timesteps = 1
    one_min = 0.016666666667
    n_mins = 5

    sampling_time = n_mins*one_min
    delta_mode = False
    tmax = int((24*60)/n_mins) # set this to 24 hours
    #tmax = 10
    print('tmax: ', tmax)
    n_episodes = 30
    train_times = []
    train_rewards = []
    test_times = []
    test_rewards = []
    pop_scaling = 100000
    print(reward_func)
    env = ChemostatEnv(param_path, reward_func, sampling_time, pop_scaling)
    print(env.action_to_Cin(0))
    print(env.action_to_Cin(1))
    print(env.action_to_Cin(2))
    print(env.action_to_Cin(3))
    print(env.reward_func)
    print('REWARD:----------------------------------------------', env.reward_func(np.array([20000, 30000]), None, None))
    print('REWARD:----------------------------------------------', env.reward_func(np.array([30000, 20000]), None, None))
    print(env.reward_func)
    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])
    #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/new_target/repeat9/saved_network.h5')
    #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/results/100eps/training_on_random/saved_network.h5')
    train_trajectorys = []
    for i in range(n_episodes):
        print('EPISODE: ', i)
        print('train: ')
        # training EPISODE
        #explore_rate = 0
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        #explore_rate = 1
        print(explore_rate)
        env.reset()
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_trajectorys.append(train_trajectory)
        train_times.append(len(train_trajectory))
        train_rewards.append(train_r)
        train_actions = np.array(agent.actions)

        values = np.array(agent.values)
        env.plot_trajectory([0,1])

        print('reward: ', train_r)
        '''
        plt.figure()
        for i in range(4):
            plt.plot(values[:, i], label = 'action ' + str(i))
        plt.legend()

        plt.figure()

        plt.plot(agent.single_ep_reward)

        env.plot_trajectory([0,1])

        plt.show()
        '''

        print()

    os.makedirs(save_path, exist_ok = True)

    # use trained policy on env with smaller smaplingn time
    #sampling_time = 0.1

    np.save(save_path + '/train_trajectories.npy', np.array(train_trajectorys))

    train_rewards = np.array(train_rewards)

    train_times = np.array(train_times)



    np.save(save_path + '/train_rewards.npy', train_rewards)

    np.save(save_path + '/train_times.npy', train_times)

    agent.save_network(save_path)

    plt.figure()
    plt.plot(train_times)
    plt.xlabel('Timestep')
    plt.ylabel('Timesteps until terminal state')
    plt.savefig(save_path + '/train_times.png', dpi = 600)


    env.plot_trajectory([0,1]) # the last test_trajectory
    plt.savefig(save_path + '/test_populations.png')
    np.save(save_path + '/test_trajectory.npy', env.sSol)


    # plot the last train trajectory
    plt.figure()
    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.savefig(save_path + '/train_populations.png', dpi = 600)
    np.save(save_path + '/train_trajectory.npy', train_trajectory)


    plt.figure()
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')

    values = np.array(agent.values)
    plt.figure()
    for i in range(4):
        plt.plot(values[:, i], label = 'action ' + str(i))
    plt.legend()

    plt.savefig(save_path + '/values.png', dpi = 600)



    values = np.array(agent.values)
    np.save(save_path + '/values.npy', values)
    np.save(save_path + '/actions.npy', agent.actions)

    # test trained policy with smaller time step


if __name__ == '__main__':
    entry()
