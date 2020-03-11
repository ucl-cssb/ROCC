from chemostat_envs import *

import yaml
import matplotlib.pyplot as plt

def fig_6_reward_function(state, action, next_state):

    N1_targ = 250
    N2_targ = 550
    targ = np.array([N1_targ, N2_targ])

    SSE = sum((state-targ)**2)

    reward = (1 - SSE/(sum(targ**2)))/10
    done = False
    if reward < 0:
        print('Reward smaller than 0: ', reward)

    if any(state < 10):
        reward = - 1
        done = True

    return reward, done

def no_LV_reward_function_new_target(state, action, next_state):

    N1_targ = 15000
    N2_targ = 25000
    targ = np.array([N1_targ, N2_targ])
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 100):
        reward = - 1
        done = True

    return reward, done

def test_trajectory():
    param_file = '/Users/ntreloar/Desktop/Projects/summer/chemostat_env/parameter_files/smaller_target_good_ICs_no_LV.yaml'


    update_timesteps = 1
    one_min = 0.016666666667
    n_mins = 10
    sampling_time = n_mins*one_min
    env = ChemostatEnv(param_file, sampling_time, update_timesteps, False)
    rew = 0

    tmax = int((24*60)/n_mins) # set this to 24 hours
    actions = []
    for i in range(tmax):

        if i%1 == 0:
            a = np.random.choice(range(4))
        else:
            a = 3

        state = env.get_state()


        a = np.random.choice(range(4))
        #a = 3
        #print(a)
        '''
        a = 3
        if i == 400:
            a = 2

        if i == 500:
            a = 1
        '''
        #a = 2


        r, done = no_LV_reward_function_new_target(state, None, None)
        print(r)
        rew += r
        env.step(a)
        if done:
            break

        actions.append(a)
    print(actions)

    env.plot_trajectory([2,3,4])
    env.plot_trajectory([0,1])
    plt.show()

    print(rew)
if __name__ == '__main__':
    test_trajectory()
