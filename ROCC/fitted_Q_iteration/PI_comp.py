import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# file path for fitted_Q_agents
FQ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FQ_DIR)

# file path for chemostat_env
C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_DIR = os.path.join(C_DIR, 'chemostat_env')
sys.path.append(C_DIR)


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
    repeat = int(arguments.repeat)

    save_path = os.path.join(arguments.save_path, 'repeat' + str(repeat))
    print(save_path)
    run_test(save_path)

def get_sum_squared_error(target, trajectory):
    target = np.array(target)

    print(trajectory.shape)
    Ns = trajectory[:,0:2]
    SSE = sum(sum(np.absolute(Ns - target)))
    return SSE


def run_test(save_path):
    param_path = os.path.join(C_DIR, 'parameter_files/double_aux.yaml')


    og_save = save_path
    n_episodes = 30
    one_min = 0.016666666667

    pop_scaling = 100000
    SSEs = []
    #for n_mins in :

    for n_mins in [1,2,3,4,5,10,20,30,40,50,60]:
        save_path = og_save + '/'+str(n_mins)+'_minutes'
        os.makedirs(save_path, exist_ok = True)
        sampling_time = n_mins*one_min
        tmax = int((24*60)/n_mins)
        times = []
        rewards = []

        env = ChemostatEnv(param_path, reward_func, sampling_time,  pop_scaling)

        agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_controlled_species*env.num_Cin_states])

        #agent.learn_heuristic()
        #os.makedirs(save_path + '/after_heuristic', exist_ok = True)
        #agent.save_network(save_path + '/after_heuristic')
        #agent.saved = save_path + '/after_heuristic'
        # generate data, need to turn update Q off for this
        for i in range(n_episodes):
            print('EPISODE: ', i)

            # training EPISODE
            #explore_rate = 0
            #explore_rate = agent.get_rate(i, 0, 1, 2.5)
            explore_rate = 1
            print(explore_rate)

            env.reset()
            #env.state = (np.random.uniform(-0.5, 0.5), 0, np.random.uniform(-0.5, 0.5), 0)
            trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train= False, remember = True)
            os.makedirs(save_path + '/episode' + str(i), exist_ok = True)
            #agent.save_network(save_path + '/episode' + str(i))


        print('number of training points: ', len(trajectory))


        # train iteratively on data
        train_rs = []
        losses = []
        min_SSE = 0
        for i in range(20):
            print('EPISODE: ', i)
            history = agent.fitted_Q_update()
            print()

            explore_rate = 0
            env.reset()
            trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train = False, remember = False)

            SSE = get_sum_squared_error([20000,30000], trajectory)
            print(SSE)
            if i == 0 or (SSE < min_SSE and len(trajectory) >= tmax):
                min_SSE = SSE
                np.save(save_path + '/trajectory_' + str(SSE), trajectory)
                env.plot_trajectory([0,1])
                plt.savefig(save_path +'/trajectory_' + str(SSE) +'.png')
                os.makedirs(save_path + '/episode' + str(i), exist_ok = True)
                #agent.save_network(save_path + '/episode' + str(i))

            losses.append(history.history['loss'])
            train_rs.append(train_r)

        plt.figure()
        plt.plot(losses)
        plt.figure()
        plt.plot(train_rs)
        print(train_r)

        explore_rate = 0
        env.reset()
        trajectory, train_r = agent.run_episode(env, explore_rate, tmax, train= False)


        rewards = np.array(rewards)

        agent.save_results(save_path)
        #agent.save_network(save_path)


        env.plot_trajectory([0,1])
        plt.savefig(save_path + '/final_trajectory' + str(SSE)+'.png')
        np.save(save_path + '/final_trajectory' + str(SSE)+'.npy', env.sSol)

        plt.figure()
        plt.plot(train_rs)
        plt.savefig(save_path + '/episode_rewards.png')

if __name__ == '__main__':
    entry()
