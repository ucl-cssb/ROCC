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
from reward_func import *



import yaml
import matplotlib.pyplot as plt

def test_trajectory():

    param_file = os.path.join(C_DIR, 'parameter_files/double_aux.yaml')


    update_timesteps = 1
    one_min = 0.016666666667
    sampling_time = one_min*10
    env = ChemostatEnv(param_file, reward_func, sampling_time, 1000)
    rew = 0

    actions = []
    for i in range(1000):
        a = np.random.choice(range(4))

        a = 3
        '''
        a = 3
        if i == 400:
            a = 2

        if i == 500:
            a = 1
        '''
        #a = 2

        state = env.get_state()
        '''
        a = 0

        if state[0] < 15000:
            a = 2
        elif state[1] < 25000:
            a = 1
        if state[0] < 15000 and state[1] < 25000:
            a = 3
        '''
        r, done = no_LV_reward_function_new_target(state, None, None)
        print(r)
        rew += r
        env.step(a)
        if done:
            break

        actions.append(a)
    print(actions)

    env.plot_trajectory([0,1])
    env.plot_trajectory([2,3,4])
    plt.show()

    print(rew)
if __name__ == '__main__':
    test_trajectory()
