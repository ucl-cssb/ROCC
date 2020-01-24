import numpy as np


def reward_func(state, action, next_state):

    N1_targ = 20000
    N2_targ = 30000
    targ = np.array([N1_targ, N2_targ])
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 1000):
        reward = - 1
        done = True

    return reward, done


def flipped_reward_func(state, action, next_state):

    N1_targ = 30000
    N2_targ = 20000
    targ = np.array([N1_targ, N2_targ])
    SE = sum(np.abs(state-targ))

    reward = (1 - sum(SE/targ)/2)/10
    done = False


    if any(state < 1000):
        reward = - 1
        done = True

    return reward, done
