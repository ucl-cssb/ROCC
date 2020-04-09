import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.colors as colors
from ROCC import *


def run_test(save_path):
    P_DIR = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ROCC'), 'chemostat_env'), 'parameter_files')
    print(P_DIR)
    param_path = os.path.join(P_DIR, 'double_aux.yaml')
    one_min = 0.016666666667
    n_mins = 5

    sampling_time = n_mins*one_min

    tmax = int((24*60)/n_mins) # set this to 24 hours
    #tmax = 10
    print('tmax: ', tmax)
    n_episodes = 30
    train_times = []
    train_returns = []

    pop_scaling = 100000
    env = ChemostatEnv(param_path, reward_func, sampling_time,  pop_scaling)

    agent = KerasFittedQAgent(layer_sizes  = [env.num_controlled_species,20,20,env.num_Cin_states**env.num_controlled_species])

    train_trajectorys = []
    for i in range(n_episodes):
        print('EPISODE: ', i)
        # training EPISODE
        explore_rate = agent.get_rate(i, 0, 1, n_episodes/10)
        print('explore rate: ', explore_rate)
        env.reset()
        train_trajectory, train_r = agent.run_episode(env, explore_rate, tmax)
        train_trajectorys.append(train_trajectory)
        train_times.append(len(train_trajectory))
        train_returns.append(train_r)
        print('return: ', train_r)


    os.makedirs(save_path, exist_ok = True)

    train_returns = np.array(train_returns)

    train_times = np.array(train_times)


    # plot the last train trajectory
    plt.figure()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Population ($10^6 cells L^{-1}$)')
    plt.hlines([20000, 30000], 0, 288*5, color = 'g', label = 'target')
    plt.xlim(left = 0)

    xSol = np.array(train_trajectory)
    for i in [0,1]:
        plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0]))*5, xSol[:,i], label = env.labels[i])
    plt.legend()
    plt.savefig(save_path + '/train_populations.png')
    np.save(save_path + '/train_trajectory.npy', train_trajectory)

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(train_returns)
    plt.savefig(save_path + '/train_returns.png')
    np.save(save_path + '/train_returns.npy', train_returns)

    N = 200
    state_action = np.zeros((N,N))
    value_function = np.zeros((N,N))
    visited_mask = np.zeros((N,N))
    min_pop = 0
    max_pop = 50000



    for i in range(0, N):
        for j in range(0, N):

            values = agent.predict(np.array([(min_pop + i*(max_pop-min_pop)/N)/pop_scaling, (min_pop + j*(max_pop-min_pop)/N)/pop_scaling]))

            action = np.argmax(values)

            value_function[i,j] = max(values)
            state_action[i,j] = action + 1


    ''' PLOT HEATMAP OF ACTIONS'''


    N=1
    data = state_action.T - 1

    plt.figure(figsize = (10,6))
    cmap = colors.ListedColormap([ 'white','blue','orange','brown'])
    bounds=[ -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    heatmap = plt.pcolor(np.array(data), cmap=cmap, norm=norm)
    cbar = plt.colorbar(heatmap)

    cbar.set_ticks([heatmap.colorbar.vmin + t*(heatmap.colorbar.vmax-heatmap.colorbar.vmin) + (heatmap.colorbar.vmax-heatmap.colorbar.vmin)/10  for t in cbar.ax.get_yticks()])
    cbar.ax.set_yticklabels(['No nutrient', 'Arginine', 'Tryptophan', 'Arginine and\ntryptophan'])

    axis_tick_interval = 40
    plt.xticks(np.arange(0, N+1, axis_tick_interval), (min_pop + np.arange(0, N+1, axis_tick_interval)*(max_pop-min_pop)/N))
    plt.yticks(np.arange(0, N+1, axis_tick_interval),(min_pop + np.arange(0, N+1, axis_tick_interval)*(max_pop-min_pop)/N))


    plt.xlabel('N1 state')
    plt.ylabel('N2 state')
    plt.savefig(save_path + '/state_action.png', dpi = 600)
    np.save(save_path + '/state_action.npy', np.array(state_action))

    ''' PLOT VALUE FUNCTION'''

    data = value_function.T

    plt.figure(figsize = (10,6))

    heatmap = plt.pcolor(np.array(data), cmap = 'plasma')
    cbar = plt.colorbar(heatmap)
    print(heatmap.colorbar.vmin, heatmap.colorbar.vmax)

    #cbar.set_ticks([heatmap.colorbar.vmin + t*(heatmap.colorbar.vmax-heatmap.colorbar.vmin) + (heatmap.colorbar.vmax-heatmap.colorbar.vmin)/10  for t in cbar.ax.get_yticks()])


    plt.xticks(np.arange(0, N+1, axis_tick_interval), (min_pop + np.arange(0, N+1, axis_tick_interval)*max_pop/N))
    plt.yticks(np.arange(0, N+1, axis_tick_interval),  (min_pop + np.arange(0, N+1, axis_tick_interval)*max_pop/N))


    plt.xlabel('N1 state')
    plt.ylabel('N2 state')
    plt.savefig(save_path + '/value_func.png', dpi = 600)
    np.save(save_path + '/value_func.npy', np.array(value_function))


if __name__ == '__main__':
    run_test('./double_aux_example_results')
