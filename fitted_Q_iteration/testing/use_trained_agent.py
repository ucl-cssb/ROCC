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


param_path = os.path.join(C_DIR, 'parameter_files/double_aux.yaml')

sampling_times = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]

for sampling_time in sampling_times:
    print(sampling_time)


    save_path = 'use_trained_agent/sampling_time' + str(sampling_time)

    tmax = int(1000/sampling_time)
    explore_rate = 0
    pop_scaling = 100000
    env = ChemostatEnv(param_path, reward_func, sampling_time, pop_scaling)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])
    agent.predict(np.array([0,0]))

    agent.load_network('/Users/ntreloar/Desktop/Projects/summer/fitted_Q_iteration/chemostat/double_aux/new_target/repeat9/saved_network.h5')
    #agent.save_network_tensorflow(os.path.dirname(os.path.abspath(__file__)) + '/100eps/training_on_random/')
    #agent.load_network_tensorflow('/Users/Neythen/Desktop/summer/fitted_Q_iteration/chemostat/100eps/training_on_random')

    trajectory = agent.run_online_episode(env, explore_rate, tmax, train = False)
    test_r = np.array([t[2] for t in trajectory])
    test_a = np.array([t[1] for t in trajectory])
    values = np.array(agent.values)


os.makedirs(save_path, exist_ok = True)


plt.figure()
plt.plot(np.linspace(0, 1000, 1000/sampling_time+1), env.sSol[:,0], label = 'N1')
plt.plot(np.linspace(0, 1000, 1000/sampling_time+1), env.sSol[:,1], label = 'N2')
plt.xlabel('Time (hours)')
plt.ylabel('Population ($10^6 cell L^{-1}$)')
plt.hlines([250, 700], 0, 1000, color = 'g')
plt.ylim(bottom = 0)
plt.savefig(save_path + '/populations.png')
np.save(save_path + '/trajectory.npy', env.sSol)


plt.figure()
plt.plot(test_r)
np.save(save_path + '/rewards.npy', test_r)
plt.savefig(save_path + '/rewards.png')


plt.figure()
plt.plot(test_a)
np.save(save_path + '/actions.npy', test_a)
plt.savefig(save_path + '/actions.png')

plt.figure()
for i in range(4):
    plt.plot(values[:, i], label = 'action ' + str(i))
plt.legend()

plt.savefig(save_path + '/values.png')

plt.figure()
plt.plot(test_a)
np.save(save_path + '/actions.npy', test_a)
plt.savefig(save_path + '/actions.png')
