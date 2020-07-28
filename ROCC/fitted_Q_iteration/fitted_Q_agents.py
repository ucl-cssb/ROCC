import sys
import os
import numpy as np
import tensorflow as tf
import math
import random
import time
from tensorflow import keras


import matplotlib.pyplot as plt

class FittedQAgent():

    '''
    abstract class for the Torch and Keras implimentations, dont use directly

    '''
    def get_action(self, state, explore_rate):
        '''
        Choses action based on enivormental state, explore rate and current value estimates

        Parameters:
            state: environmental state
            explore_rate
        Returns:
            action
        '''

        if np.random.random() < explore_rate:
            action = np.random.choice(range(self.layer_sizes[-1]))

        else:
            values = self.predict(state)
            self.values.append(values)
            action = np.argmax(values)
            self.actions.append(action)

        assert action < self.n_actions, 'Invalid action'
        return action

    def get_inputs_targets(self):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''


        targets = []


        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        # iterate over all exprienc in memory and create fitted Q targets
        for trajectory in self.memory:

            for transition in trajectory:
                state, action, reward, next_state, done = transition

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

        states = np.array(states)
        next_states = np.array(next_states, dtype=np.float64)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # construct target
        values = self.predict(states)
        next_values = self.predict(next_states)

        #update the value for the taken action using cost function and current Q
        for i in range(len(next_states)):
            # print(actions[i], rewards[i])
            if dones[i]:

                values[i, actions[i]] = rewards[i]
            else:
                values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_values[i])

        # shuffle inputs and target for IID
        inputs, targets  = np.array(states), np.array(values)
        randomize = np.arange(len(inputs))
        np.random.shuffle(randomize)
        inputs = inputs[randomize]
        targets = targets[randomize]
        return inputs, targets

    def fitted_Q_update(self, inputs = None, targets = None):
        '''
        Uses a set of inputs and targets to update the Q network
        '''
        t = time.time()
        if inputs is None and targets is None:
            inputs, targets = self.get_inputs_targets()


        t = time.time()
        self.reset_weights()


        t = time.time()
        history = self.fit(inputs, targets)
        return history

    def run_episode(self, env, explore_rate, tmax, train = True, remember = True):
        '''
        Runs one fitted Q episode

        Parameters:
         env: the environment to train on and control
         explore_rate: explore rate for this episodes
         tmax: number of timesteps in the episode
         train: does the agent learn?
         remember: does the agent store eperience in its memory?

        Returns:
            env.sSol: time evolution of environmental states
            episode reward: total reward for this episode
        '''
        # run trajectory with current policy and add to memory
        trajectory = []
        actions = []

        state = env.get_state()
        episode_reward = 0
        self.single_ep_reward = []
        for i in range(tmax):

            action = self.get_action(state, explore_rate)

            actions.append(action)

            next_state, reward, done, info = env.step(action)


            assert len(next_state) == self.state_size, 'env return state of wrong size'

            self.single_ep_reward.append(reward)
            if done:
                print(reward)

            # scale populations

            transition = (state, action, reward, next_state, done)
            state = next_state
            trajectory.append(transition)
            episode_reward += reward

            if done: break


        if remember: # store this episode
            self.memory.append(trajectory)

        if train: # train the agent

            self.actions = actions
            self.episode_lengths.append(i)
            self.episode_rewards.append(episode_reward)


            if len(self.memory[0]) * len(self.memory) < 100:
                n_iters = 4
            elif len(self.memory[0]) * len(self.memory) < 200:
                n_iters = 5
            else:
                n_iters = 10

            for _ in range(n_iters):

                self.fitted_Q_update()

        return env.sSol, episode_reward

    def neural_fitted_Q(self, env, n_episodes, tmax):
        '''
        runs a whole neural fitted Q experiment

        Parameters:
            env: environment to train on
            n_episodes: number of episodes
            tmax: timesteps in each episode
        '''

        times = []
        for i in range(n_episodes):
            print()
            print('EPISODE', i)

            explore_rate = self.get_rate(i, 0, 1, 2.5)

            print('explore_rate:', explore_rate)
            env.reset()
            trajectory, reward = self.run_episode(env, explore_rate, tmax)

            time = len(trajectory)
            print('Time: ', time)
            times.append(time)

        print(times)

    def plot_rewards(self):
        '''
        Plots the total reward gained in each episode on a matplotlib figure
        '''
        plt.figure(figsize = (16.0,12.0))

        plt.plot(self.episode_rewards)

    def save_results(self, save_path):
        '''
        saves numpy arrays of results of training
        '''
        np.save(save_path + '/survival_times', self.episode_lengths)
        np.save(save_path + '/episode_rewards', self.episode_rewards)

    def get_rate(self, episode, MIN_LEARNING_RATE,  MAX_LEARNING_RATE, denominator):
        '''
        Calculates the logarithmically decreasing explore or learning rate

        Parameters:
            episode: the current episode
            MIN_LEARNING_RATE: the minimum possible step size
            MAX_LEARNING_RATE: maximum step size
            denominator: controls the rate of decay of the step size
        Returns:
            step_size: the Q-learning step size
        '''

        # input validation
        if not 0 <= MIN_LEARNING_RATE <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= MAX_LEARNING_RATE <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < denominator:
            raise ValueError("denominator needs to be above 0")

        rate = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, 1.0 - math.log10((episode+1)/denominator)))

        return rate


class KerasFittedQAgent(FittedQAgent):
    '''
    Implementation of the neural network using keras
    '''
    def __init__(self, layer_sizes = [2,20,20,4]):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.network = self.initialise_network(layer_sizes)
        self.gamma = 0.99
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.episode_lengths = []
        self.episode_rewards = []
        self.single_ep_reward = []
        self.total_loss = 0
        self.values = []
        self.actions = []

    def initialise_network(self, layer_sizes):

        '''
        Creates Q network for value function approximation
        '''

        tf.keras.backend.clear_session()
        initialiser = keras.initializers.RandomUniform(minval = -0.5, maxval = 0.5, seed = None)
        positive_initialiser = keras.initializers.RandomUniform(minval = 0., maxval = 0.35, seed = None)
        regulariser = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        network = keras.Sequential([
            keras.layers.InputLayer([layer_sizes[0]]),
            keras.layers.Dense(layer_sizes[1], activation = tf.nn.relu),
            keras.layers.Dense(layer_sizes[2], activation = tf.nn.relu),
            keras.layers.Dense(layer_sizes[3]) # linear output layer
        ])

        network.compile(optimizer = 'adam', loss = 'mean_squared_error') # TRY DIFFERENT OPTIMISERS
        return network

    def predict(self, state):
        '''
        Predicts value estimates for each action base on currrent states
        '''

        return self.network.predict(state.reshape(-1,self.layer_sizes[0]))

    def fit(self, inputs, targets):
        '''
        trains the Q network on a set of inputs and targets
        '''
        history = self.network.fit(inputs, targets,  epochs = 150, verbose = False)
        return history

    def reset_weights(self):
        '''
        Reinitialises weights to random values
        '''
        #sess = tf.keras.backend.get_session()
        #sess.run(tf.global_variables_initializer())
        t = time.time()
        self.network = self.initialise_network(self.layer_sizes)

    def save_network(self, save_path):
        '''
        Saves current network weights
        '''
        self.network.save(save_path + '/saved_network.h5')

    def save_network_tensorflow(self, save_path):
        '''
        Saves current network weights using pure tensorflow, kerassaver seems to crash sometimes
        '''
        saver = tf.train.Saver()
        sess = tf.keras.backend.get_session()
        path = saver.save(sess, save_path + "/saved/model.cpkt")

    def load_network_tensorflow(self, save_path):
        '''
        Loads network weights from file using pure tensorflow, kerassaver seems to crash sometimes
        '''

        saver = tf.train.Saver()

        sess = tf.keras.backend.get_session()
        saver.restore(sess, save_path +"/saved/model.cpkt")

    def load_network(self, load_path):
        '''
        Loads network weights from file
        '''
        try:
            self.network = keras.models.load_model(load_path + '/saved_network.h5') # sometimes this crashes, apparently a bug in keras
        except:
            self.network.load_weights(load_path + '/saved_network.h5') # this requires model to be initialised exactly the same
