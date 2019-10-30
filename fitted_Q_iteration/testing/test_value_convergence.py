import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
from reward_func import *

np.set_printoptions(precision = 16)



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

def run_test(save_path):
    param_path = os.path.join(C_DIR, 'parameter_files/double_aux.yaml')

    one_min = 0.016666666667
    n_mins = 5
    sampling_time = n_mins*one_min

    tmax = 1000
    pop_scaling = 100000

    env = ChemostatEnv(param_path, reward_func, sampling_time, pop_scaling)
    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])
    state = env.reset()


    trajectory = []
    actions = []
    rewards = []

    n_repeats = 100

    all_value_SSEs = []
    for repeat in range(1,n_repeats+1):
        print('----------------------------------')
        print('REPEAT', repeat)

        n_mins = 5
        sampling_time = n_mins*one_min
        delta_mode = False
        tmax = int((24*60)/n_mins) # set this to 24 hours
        tmax = 1000
        print('tmax: ', tmax)
        train_times = []
        train_rewards = []
        test_times = []
        test_rewards = []

        env = ChemostatEnv(param_path, reward_func, sampling_time,  pop_scaling)

        explore_rate = 1
        all_pred_rewards = []
        all_actual_rewards = []

        os.makedirs(save_path, exist_ok = True)
        agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])
        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_actions = agent.actions
        n_iters = 10
        agent.reset_weights()

        value_SSEs = []

        true_values = [agent.single_ep_reward[-1]]


        for i in range(2,len(agent.single_ep_reward)+1):
            true_values.insert(0, agent.single_ep_reward[-i] + true_values[0] * agent.gamma)




        for iter in range(1,n_iters+1):
            print()
            print('ITER: ' + str(iter), '------------------------------------')
            history = agent.fitted_Q_update()

            print('losses: ', history.history['loss'][0], history.history['loss'][-1])
            values = []

            training_pred = []
            for i in range(len(train_trajectory[:,0:2])-1):
                values = agent.predict(train_trajectory[i,0:2]/100000) # appends to agent.values
                training_pred.append(values[train_actions[i]])


            # get the change from last value function to measure convergence
            #print(all_pred_rewards)



            SSE = sum((np.array(true_values)[0:-300] - np.array(training_pred)[0:-300])**2)
            print('sse:', SSE)
            value_SSEs.append(SSE)

        plt.figure()
        plt.plot(true_values[0:-300], label = 'true')
        plt.plot(training_pred[0:-300], label = 'pred')
        plt.legend()


        print(value_SSEs)
        plt.figure()
        plt.plot(value_SSEs)
        plt.title('SSEs')
            #plt.savefig(save_path + '/' + str(n_iters))
        plt.show()


        all_value_SSEs.append(value_SSEs)
    all_value_SSEs = np.array(all_value_SSEs)
    print(all_value_SSEs.shape)
    np.save(save_path + 'all_value_SSEs.npy', all_value_SSEs)
    print(all_value_SSEs)



    '''
    train_trajectory, train_r = agent.run_episode(env, 0, tmax)
    print(train_r)
    # plot the last train trajectory
    plt.figure()
    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.show()
    '''




if __name__ == '__main__':
    entry()
