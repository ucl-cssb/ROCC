"""
Scaled reproduction of Figure 2 of Treloar et al. (2020).

One seed per invocation. Everything needed to trace a result back to the run
that produced it is written into the output directory.

    python scripts/run_fig2.py --seed 0 --episodes 20 --out results/raw

Original experiment: 20 agents x 30 episodes, mean final return 27.4 (SD 0.33),
theoretical maximum 28.8.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from ROCC import ChemostatEnv, KerasFittedQAgent, reward_func

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_FILE = os.path.join(REPO, 'ROCC', 'chemostat_env', 'parameter_files', 'double_aux.yaml')

ONE_MIN = 0.016666666667
POP_SCALING = 100000


def git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def git_dirty():
    try:
        out = subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain'])
        return bool(out.strip())
    except Exception:
        return None


def cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--sampling-minutes', type=int, default=5)
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'raw'))
    args = ap.parse_args()

    run_id = 'fig2_seed{}_ep{}'.format(args.seed, args.episodes)
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Seed every source of randomness before anything is constructed.
    # tf's graph-level seed is set inside the agent, after clear_session().
    random.seed(args.seed)
    np.random.seed(args.seed)

    sampling_time = args.sampling_minutes * ONE_MIN
    tmax = int((24 * 60) / args.sampling_minutes)   # 288 steps = 24 h at 5 min

    env = ChemostatEnv(PARAM_FILE, reward_func, sampling_time, POP_SCALING)
    agent = KerasFittedQAgent(
        layer_sizes=[env.num_controlled_species, 20, 20,
                     env.num_Cin_states ** env.num_controlled_species],
        seed=args.seed,
    )

    returns, explore_rates, episode_seconds = [], [], []
    trajectory = None
    t_start = time.time()

    for ep in range(args.episodes):
        t_ep = time.time()
        explore_rate = agent.get_rate(ep, 0, 1, args.episodes / 10)
        env.reset()
        trajectory, ep_return = agent.run_episode(env, explore_rate, tmax)

        returns.append(ep_return)
        explore_rates.append(explore_rate)
        episode_seconds.append(time.time() - t_ep)
        print('episode {:3d}  explore {:.4f}  return {:8.3f}  {:6.1f}s'.format(
            ep, explore_rate, ep_return, episode_seconds[-1]), flush=True)

    total_seconds = time.time() - t_start

    np.save(os.path.join(out_dir, 'returns.npy'), np.array(returns))
    np.save(os.path.join(out_dir, 'explore_rates.npy'), np.array(explore_rates))
    np.save(os.path.join(out_dir, 'episode_seconds.npy'), np.array(episode_seconds))
    np.save(os.path.join(out_dir, 'final_trajectory.npy'), np.array(trajectory))
    agent.save_network(out_dir)

    config = {
        'run_id': run_id,
        'seed': args.seed,
        'episodes': args.episodes,
        'sampling_minutes': args.sampling_minutes,
        'timesteps_per_episode': tmax,
        'discount_gamma': agent.gamma,
        'layer_sizes': agent.layer_sizes,
        'param_file': os.path.relpath(PARAM_FILE, REPO),
        'targets_cells_per_L': [20e9, 30e9],
        'theoretical_max_return': round(tmax * 0.1, 3),
        'final_return': returns[-1],
        'best_return': max(returns),
        'total_seconds': round(total_seconds, 1),
        'mean_episode_seconds': round(float(np.mean(episode_seconds)), 1),
        'git_commit': git_commit(),
        'git_dirty': git_dirty(),
        'python': sys.version.split()[0],
        'tensorflow': tf.__version__,
        'numpy': np.__version__,
        'platform': platform.platform(),
        'cpu': cpu_model(),
        'cpu_count': os.cpu_count(),
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('\nwrote {}'.format(out_dir))
    print('final return {:.3f} of {:.1f} max, {:.1f} min total'.format(
        returns[-1], config['theoretical_max_return'], total_seconds / 60))


if __name__ == '__main__':
    main()
