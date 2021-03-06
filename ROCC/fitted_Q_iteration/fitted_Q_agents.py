import sys
import os
import numpy as np
import tensorflow as tf
import math
import random

from tensorflow import keras


'''
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
'''
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


        assert action < self.n_actions, 'Invalid action'
        return action

    def get_inputs_targets(self):
        '''
        gets fitted Q inputs and calculates targets for training the Q-network for episodic training
        '''

        inputs = []
        targets = []

        # DO THIS WITH NUMPY TO MAKE IT FASTER
        for trajectory in self.memory:

            for transition in trajectory:
                # CHEKC TARGET IS BUILT CORRECTLY

                state, action, cost, next_state, done = transition
                inputs.append(state)
                # construct target
                values = self.predict(state)

                next_values = self.predict(next_state)

                assert len(values) == self.n_actions, 'neural network returning wrong number of values'
                assert len(next_values) == self.n_actions, 'neural network returning wrong number of values'

                #update the value for the taken action using cost function and current Q

                if not done:
                    values[action] = cost + self.gamma*np.max(next_values) # could introduce step size here, maybe not needed for neural agent
                else:
                    values[action] = cost

                targets.append(values)

        # shuffle inputs and target for IID
        inputs, targets  = np.array(inputs), np.array(targets)



        randomize = np.arange(len(inputs))
        np.random.shuffle(randomize)
        inputs = inputs[randomize]
        targets = targets[randomize]


        assert inputs.shape[1] == self.state_size, 'inputs to network wrong size'
        assert targets.shape[1] == self.n_actions, 'targets for network wrong size'
        return inputs, targets

    def fitted_Q_update(self, inputs = None, targets = None):
        '''
        Uses a set of inputs and targets to update the Q network
        '''

        if inputs is None and targets is None:
            inputs, targets = self.get_inputs_targets()

        #
        #tf.initialize_all_variables() # resinitialise netowrk without adding to tensorflow graph
        # try RMSprop and adam and maybe some from here https://arxiv.org/abs/1609.04747
        self.reset_weights()

        history = self.fit(inputs, targets)
        #print('losses: ', history.history['loss'][0], history.history['loss'][-1])
        return history

    def run_episode(self, env, explore_rate, tmax, train = True, remember = True):
        '''
        Runs one fitted Q episode

        Parameters:
         env: the enirovment to train on and control
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
        #self.values = []
        state = env.get_state()
        episode_reward = 0
        self.single_ep_reward = []
        for i in range(tmax):

            action = self.get_action(state, explore_rate)

            actions.append(action)

            next_state, reward, done, info = env.step(action)

            #cost = -cost # as cartpole default returns a reward
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


        if remember:
            self.memory.append(trajectory)

        if train:

            self.actions = actions
            self.episode_lengths.append(i)
            self.episode_rewards.append(episode_reward)


            if len(self.memory[0]) * len(self.memory) < 100:
                #n_iters = 4
                n_iters = 4
            elif len(self.memory[0]) * len(self.memory) < 200:
                #n_iters = 5
                n_iters = 5
            else:
                n_iters = 10

            #n_iters = 0
            for _ in range(n_iters):

                self.fitted_Q_update()

        #env.plot_trajectory()
        #plt.show()
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


            # CONSTANT EXPLORE RATE OF 0.1 worked well
            explore_rate = self.get_rate(i, 0, 1, 2.5)
            #explore_rate = 0.1
            #explore_rate = 0
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
    def __init__(self, layer_sizes = [2,20,20,4]):
        self.memory = []
        self.layer_sizes = layer_sizes
        self.network = self.initialise_network(layer_sizes)
        self.gamma = 0.9
        self.state_size = layer_sizes[0]
        self.n_actions = layer_sizes[-1]
        self.episode_lengths = []
        self.episode_rewards = []
        self.single_ep_reward = []
        self.total_loss = 0
        self.values = []


    def initialise_network(self, layer_sizes):

        '''
        Creates Q network
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

        return self.network.predict(state.reshape(1,-1))[0]

    def fit(self, inputs, targets):
        '''
        trains the Q network on a set of inputs and targets
        '''
        history = self.network.fit(inputs, targets,  epochs = 300, verbose = 0) # TRY DIFFERENT EPOCHS
        return history

    def reset_weights(model):
        '''
        Reinitialises weights to random values
        '''
        sess = tf.keras.backend.get_session()
        sess.run(tf.global_variables_initializer())

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


    def load_network(self, load_path): #tested
        '''
        Loads network weights from file
        '''
        try:
            self.network = keras.models.load_model(load_path + '/saved_network.h5') # sometimes this crashes, apparently a bug in keras
        except:
            self.network.load_weights(load_path + '/saved_network.h5') # this requires model to be initialised exactly the same
