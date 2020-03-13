import sys
import os
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

    n_mins = 5
    n_envs = 5
    one_min = 0.016666666667
    sampling_time = n_mins*one_min

    tmax = 10
    print('tmax: ', tmax)
    n_episodes = 29
    train_rewards = []

    pop_scaling = 100000
    os.makedirs(save_path, exist_ok = True)
    os.makedirs(save_path + '/after_heuristic', exist_ok = True)
    envs = [ChemostatEnv(param_path, reward_func, sampling_time,  pop_scaling) for i in range(n_envs)]

    agent = KerasFittedQAgent(layer_sizes  = [envs[0].num_controlled_species,20,20,envs[0].num_Cin_states**envs[0].num_controlled_species])

    agent.memory = []


    overall_traj = []

    for i in range(n_episodes):

        print()
        print('EPISODE: ', i)
        print('train: ')
        # training EPISODE
        #
        #agent.memory = []

        explore_rate = agent.get_rate(i, 0.05, 1., n_episodes/10)

        #explore_rate = 1
        print(explore_rate)

        for env in envs[:-1]:
            # use policy on all envs and add to memory
            train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train = False, remember = True)
            train_rewards.append(train_r)

        # only train on last env
        train_trajectory, train_r = agent.run_episode(envs[-1], explore_rate, tmax, train = True, remember = True)
        train_rewards.append(train_r)

        os.makedirs(save_path + '/episode' + str(i), exist_ok = True)
        agent.save_network(save_path + '/episode' + str(i))

        values = np.array(agent.values)

    # use trained policy on env with smaller smaplingn time
    #sampling_time = 0.1

    train_rewards = np.array(train_rewards)


    np.save(save_path + '/train_rewards.npy', train_rewards)

    agent.save_network(save_path)

    for i, env in enumerate(envs):
        env.plot_trajectory([0,1]) # the last test_trajectory
        plt.savefig(save_path + '/test_populations_' + str(i)+ '.png')
        np.save(save_path + '/test_trajectory_' + str(i)+ '.npy', env.sSol)



    plt.figure()
    plt.plot(train_rewards)
    plt.savefig(save_path + '/train_rewards.png')




if __name__ == '__main__':
    run_test('parallel_example_results')
