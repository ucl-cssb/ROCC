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

    tmax = 100
    pop_scaling = 100000



    n_repeats = 50


    for tmax in [5, 10, 20, 30, 40, 50, 100, 200, 300,500,700,1000,2000]:

        all_train_errors = []
        all_test_errors = []

        all_train_trajectories = []
        all_test_trajectories = []


        all_pred_rewards = []
        all_actual_rewards = []

        for repeat in range(1,n_repeats+1):
            print(tmax, repeat,'------------------------------------------')

            sampling_time = n_mins*one_min
            delta_mode = False
            #tmax = int((24*60)/n_mins) # set this to 24 hours

            print('tmax: ', tmax)
            train_times = []
            train_rewards = []
            test_times = []
            test_rewards = []

            env = ChemostatEnv(param_path, reward_func, sampling_time, pop_scaling)

            explore_rate = 1


            os.makedirs(save_path, exist_ok = True)
            agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])
            test_trajectory, _ = agent.run_episode(env, explore_rate, 1000)
            test_actions = agent.actions
            test_r = agent.single_ep_reward
            env.plot_trajectory([0,1])
            plt.savefig('test_trajectory.png')


            env.reset()
            agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])
            train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax) # adds to agents memory
            train_actions = agent.actions
            train_r = agent.single_ep_reward
            env.plot_trajectory([0,1])
            plt.savefig('train_trajectory.png')

            n_iters = 20
            train_errors = []
            test_errors = []

            for i in range(n_iters):
                print('ITER: ' + str(i), '------------------------------------')
                history = agent.fitted_Q_update() # fit to training data

                print('losses: ', history.history['loss'][0], history.history['loss'][-1])
                values = []

                ## GET REWARDS PREDICTED FROM TRAINING DATA
                #predict values after training
                training_pred = []
                testing_pred = []

                print(len(train_trajectory[:,0:2]))
                print(len(train_actions))

                for i in range(len(train_trajectory[:,0:2])-1):
                    values = agent.predict(train_trajectory[i,0:2]/100000) # appends to agent.values
                    training_pred.append(values[train_actions[i]])

                for i in range(len(test_trajectory[:,0:2])-1):
                    values = agent.predict(test_trajectory[i,0:2]/100000)
                    testing_pred.append(values[test_actions[i]])


                train_errors.append(sum((np.array(training_pred) - np.array(train_r))**2)/len(train_r))
                test_errors.append(sum((np.array(testing_pred) - np.array(test_r))**2)/len(test_r))

                print('pred: ', training_pred)
                print('actual: ', train_r)
                #print('error:', np.array(training_pred) - np.array(train_r))
                print('train error: ', sum((np.array(training_pred) - np.array(train_r))**2)/len(train_r))
                print('test error: ', sum((np.array(testing_pred) - np.array(test_r))**2)/len(test_r))


            '''
            plt.figure()
            plt.plot(test_r, label = 'actual reward')
            plt.plot(testing_pred, label = 'pred reward')
            plt.legend()
            plt.savefig('reward_prediction.png')
            '''
            all_pred_rewards.append(testing_pred)
            all_actual_rewards.append(test_r)




            '''
            plt.figure()
            plt.plot(train_errors, label = 'train errors')
            plt.plot(test_errors, label = 'test errors')
            plt.legend()

            plt.savefig('error_during_training.png')
            '''
            all_train_errors.append(train_errors)
            all_test_errors.append(test_errors)



            all_train_trajectories.append(train_trajectory)
            all_test_trajectories.append(test_trajectory)


        np.save(save_path + '_' +str(tmax)+'_all_pred_rewards.npy',np.array(all_pred_rewards))
        np.save(save_path +'_' +str(tmax)+'all_actual_rewards.npy',np.array(all_actual_rewards))

        print(np.array(all_pred_rewards).shape)
        print(np.array(all_actual_rewards).shape)

        np.save(save_path +'_' +str(tmax)+'all_train_errors.npy',np.array(all_train_errors))
        np.save(save_path +'_' +str(tmax)+'all_test_errors.npy',np.array(all_test_errors))

        print(np.array(all_train_errors).shape)
        print(np.array(all_test_errors).shape)


        np.save(save_path +'_' +str(tmax)+'all_train_trajectories.npy',np.array(all_test_trajectories))
        np.save(save_path +'_' +str(tmax)+'all_test_trajectories.npy',np.array(all_train_trajectories))

        print(np.array(all_test_trajectories).shape)
        print(np.array(all_train_trajectories).shape)



















if __name__ == '__main__':
    entry()
