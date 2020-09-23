import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)


from chemostat_envs import *
from fitted_Q_agents import *
from argparse import ArgumentParser


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
    repeat = int(arguments.repeat)

    save_path = os.path.join(arguments.save_path, 'repeat' + str(repeat))

    run_test(save_path)

def run_test(save_path):
    param_path = os.path.join(C_DIR, 'parameter_files/product.yaml')

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
    #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/new_target/repeat9/saved_network.h5')
    #agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/results/100eps/training_on_random/saved_network.h5')
    train_trajectorys = []
    for i in range(n_episodes):
        print('EPISODE: ', i)
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        print('\texplore_rate: ', explore_rate)
        initial_S = np.append(np.append(np.append(np.random.uniform(10000, 50000, 2), env.initial_C), env.initial_C0),env.initial_chems)
        env.reset(initial_S)
        #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
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
    plt.ylabel('Populations')
    plt.xlabel('Timestep')
    plt.savefig(save_path + '/train_populations.png')
    np.save(save_path + '/train_trajectory.npy', train_trajectory)


    plt.figure()
    plt.title('Return')
    plt.xlabel('Episode')
    plt.label('Return')
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')





if __name__ == '__main__':
    entry()
