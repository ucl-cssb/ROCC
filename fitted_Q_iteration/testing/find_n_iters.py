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
    update_timesteps = 1
    one_min = 0.016666666667
    n_mins = 1
    sampling_time = n_mins*one_min
    delta_mode = False
    tmax = 100
    pop_scaling = 100000

    env = ChemostatEnv(param_path, reward_func, sampling_time, pop_scaling)
    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])
    state = env.reset()


    trajectory = []
    actions = []
    rewards = []

    n_repeats = 1

    all_value_SSEs = []
    for repeat in range(1,n_repeats+1):

        n_mins = 30
        sampling_time = n_mins*one_min
        delta_mode = False
        tmax = int((24*60)/n_mins) # set this to 24 hours
        tmax = 1000
        print('tmax: ', tmax)
        train_times = []
        train_rewards = []
        test_times = []
        test_rewards = []

        env = ChemostatEnv(param_path, reward_func, sampling_time, pop_scaling)

        explore_rate = 1
        all_pred_rewards = []
        all_actual_rewards = []

        os.makedirs(save_path, exist_ok = True)
        agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species*update_timesteps,20,20,env.num_Cin_states**env.num_controlled_species])
        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)

        n_iters = 1
        agent.reset_weights()

        value_SSEs = []

        plt.figure()



        true_values = [agent.single_ep_reward[-1]]



        for i in range(2,len(agent.single_ep_reward)+1):
            true_values.insert(0, agent.single_ep_reward[-i] + true_values[0] * agent.gamma)


        plt.plot(true_values, label = 'actual')


        for iter in range(1,n_iters+1):
            print()
            print('ITER: ' + str(iter), '------------------------------------')
            history = agent.fitted_Q_update()

            print('losses: ', history.history['loss'][0], history.history['loss'][-1])
            values = []

            #predict values after training
            agent.values = []

            for state in train_trajectory[:,0:2]:
                agent.get_action(state/100000, 0) # appends to agent.values


            print(len(agent.values))
            print(len(agent.actions))

            values = np.array(agent.values)

            pred_rewards = []

            for i in range(len(agent.actions)):
                action_values = values[i]
                action_taken = agent.actions[i]
                pred_rewards.append(action_values[action_taken])

            # get the change from last value function to measure convergence
            #print(all_pred_rewards)

            if iter != 1:

                SSE = sum((np.array(all_pred_rewards[-1]) - np.array(pred_rewards))**2)
                print('sse:', SSE)
                value_SSEs.append(SSE)

            all_value_SSEs.append(value_SSEs)

            all_pred_rewards.append(pred_rewards)
            all_actual_rewards.append(agent.single_ep_reward)




            '''
            print('pred_rewards:', pred_rewards)
            print()
            print('single_ep_reward:', agent.single_ep_reward)
            '''

            '''
            plt.plot(agent.single_ep_reward, label = 'actual')
            plt.plot(pred_rewards, label = 'pred')
            '''

            if iter%5 == 0:

                plt.plot(pred_rewards, label = str(iter))


        plt.legend()



        values = np.array(agent.values)
        plt.figure()
        for i in range(4):
            plt.plot(values[:, i], label = 'action ' + str(i))
        plt.legend()



        plt.figure()
        plt.plot(value_SSEs)
        plt.title('SSEs')
            #plt.savefig(save_path + '/' + str(n_iters))


        env.plot_trajectory([0, 1])



        plt.show()

        np.save(save_path + '/repeat' + str(repeat) +'all_actual_r', all_actual_rewards)
        np.save(save_path + '/repeat' + str(repeat) + 'all_pred_r', all_pred_rewards)
        np.save(save_path + '/repeat' + str(repeat) + 'av_SSE', sum((np.array(all_actual_rewards) - np.array(all_pred_rewards))**2))
        #print('all_actual: ',all_actual_rewards)
        #print('all_pred:' ,all_pred_rewards)
        print(sum(sum((np.array(all_actual_rewards) - np.array(all_pred_rewards))**2)))




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
