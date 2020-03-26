
import yaml
import numpy as np
from scipy.integrate import odeint


import matplotlib.pyplot as plt

class ChemostatEnv():

    '''
    Chemostat environment that can handle an arbitrary number of bacterial strains where all are being controlled

    '''

    def __init__(self, param_file, reward_func, sampling_time, scaling):
        '''
        Parameters:
            param_file: path of a yaml file contining system parameters
            reward_func: python function used to coaculate reward: reward = reward_func(state, action, next_state)
            sampling_time: time between sampl-and-hold intervals
            scaling: population scaling to prevent neural network instability in agent, aim to have pops between 0 and 1. env returns populations/scaling to agent
        '''
        self.scaling = scaling
        f = open(param_file)
        param_dict = yaml.load(f)
        f.close()
        #self.validate_param_dict(param_dict)
        param_dict = self.convert_to_numpy(param_dict)
        self.set_params(param_dict)
        print(self.initial_N)
        self.initial_S = np.append(np.append(self.initial_N, self.initial_C), self.initial_C0)
        self.S = self.initial_S
        self.sSol = np.array(self.initial_S).reshape(1,len(self.S))
        self.labels = ['N1', 'N2', 'C1', 'C2', 'C0']
        self.Cins = []
        self.state = self.get_state()
        self.sampling_time = sampling_time
        self.reward_func = reward_func


    def convert_to_numpy(self,param_dict):
        '''
        Takes a parameter dictionary and converts the required parameters into numpy
        arrays

        Parameters:
            param_dict: the parameter dictionary
        Returns:
            param_dict: the converted parameter dictionary
        '''

        # convert all relevant parameters into numpy arrays
        param_dict['ode_params'][2], param_dict['ode_params'][3],param_dict['ode_params'][4], param_dict['ode_params'][5],  param_dict['ode_params'][6],  param_dict['ode_params'][7]  = \
            np.array(param_dict['ode_params'][2]), np.array(param_dict['ode_params'][3]), np.array(param_dict['ode_params'][4]),np.array(param_dict['ode_params'][5]), np.array(param_dict['ode_params'][6]), np.array(param_dict['ode_params'][7])


        param_dict['env_params'][3],param_dict['env_params'][4], param_dict['env_params'][5] = \
             np.array(param_dict['env_params'][3]), np.array(param_dict['env_params'][4]),np.array(param_dict['env_params'][5])

        return param_dict

    def validate_param_dict(self,param_dict):
        '''
        Performs input validation on the parameter dictionary supplied by the user
        and throws an error if parameters are invalid.

        Parameters:
            param_dict: the parameter dictionary
        '''

        # validate ode_params
        ode_params = param_dict['ode_params']

        if ode_params[0] <= 0:
            raise ValueError("C0in needs to be positive")
        if ode_params[1] <= 0:
            raise ValueError("q needs to be positive")
        if not all(y > 0 for y in ode_params[2]) or not all(y3 > 0 for y3 in ode_params[3]):
            raise ValueError("all bacterial yield constants need to be positive")
        if not all(Rmax > 0 for Rmax in ode_params[4]):
            raise ValueError("all maximum growth rates need to be positive")
        if not all(Km >= 0 for Km in ode_params[5]) or not all(Km3 >= 0 for Km3 in ode_params[6]):
            raise ValueError("all saturation constants need to be positive")


        # validate Q_params
        env_params = param_dict['env_params']
        num_species = env_params[0]

        if num_species < 0 or not isinstance(num_species, int):
            raise ValueError("num_species needs to be a positive integer")

        if env_params[1] > num_species or env_params[1] < 0 or not isinstance(env_params[1], int):
            raise ValueError("num_controlled_species needs to be a positive integer <= to num_species")

        if env_params[2] < 0 or not isinstance(num_species, int):
            raise ValueError("num_Cin_states needs to be a positive integer")
        if len(env_params[3]) != 2 or env_params[3][0] < 0 or env_params[3][0] >= env_params[3][1]:
            raise ValueError("Cin_bounds needs to be a list with two values in ascending order")


        if not all(x > 0 for x in env_params[4]):
            raise ValueError("all initial populations need to be positive")
        if not all(c > 0 for c in env_params[5]):
            raise ValueError("all initial concentrations need to be positive")
        if env_params[6] < 0:
            raise ValueError("initial C0 needs to be positive")

    def set_params(self,param_dict):
        '''
        Sets env params to those stored in a python dictionary
            Parameters:
                param_dict : ptyhon dictionary containing all params
        '''
        self.C0in, self.q, self.y, self.y0, self.umax, self.Km, self.Km0, self.A = param_dict['ode_params']
        self.num_species, self.num_controlled_species, self.num_Cin_states, self.Cin_bounds, self.initial_N, self.initial_C, self.initial_C0 = param_dict['env_params']

    def monod(self,C, C0):
        '''
        Calculates the growth rate based on the monod equation

        Parameters:
            C: the concetrations of the auxotrophic nutrients for each bacterial
                population
            C0: concentration of the common carbon source
            Rmax: array of the maximum growth rates for each bacteria
            Km: array of the saturation constants for each auxotrophic nutrient
            Km0: array of the saturation constant for the common carbon source for
                each bacterial species
        '''

        # convert to numpy

        growth_rate = ((self.umax*C)/ (self.Km + C)) * (C0/ (self.Km0 + C0))

        return growth_rate

    def sdot(self,S, t, Cin):
        '''
        Calculates and returns derivatives for the numerical solver odeint

        Parameters:
            S: current state
            t: current time
            Cin: array of the concentrations of the auxotrophic nutrients and the
                common carbon source
            params: list parameters for all the exquations
            num_species: the number of bacterial populations
        Returns:
            dsol: array of the derivatives for all state variables
        '''

        # extract variables
        N = np.array(S[:self.num_species])
        C = np.array(S[self.num_species:self.num_species+self.num_controlled_species])
        C0 = np.array(S[-1])

        R = self.monod(C, C0)

        Cin = Cin[:self.num_controlled_species]

        # calculate derivatives
        dN = N * (R.astype(float) + np.matmul(self.A, N) - self.q) # q term takes account of the dilution
        dC = self.q*(Cin - C) - (1/self.y)*R*N # sometimes dC.shape is (2,2)
        dC0 = self.q*(self.C0in - C0) - sum(1/self.y0[i]*R[i]*N[i] for i in range(self.num_species))

        if dC.shape == (2,2):
            print(q,Cin.shape,C0,C,y,R,N) #C0in

        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])
        dsol = np.append(dN, dC)
        dsol = np.append(dsol, dC0)


        return dsol

    def step(self, action):
        '''
        Performs one sampling and hold interval using the action provided by a reinforcment leraning agent

        Parameters:
            action: action chosen by agent
        Returns:
            state: scaled state to be observed by agent
            reward: reward obtained buring this sample-and-hold interval
            done: boolean value indicating whether the environment has reached a terminal state


        '''
        Cin = self.action_to_Cin(action)

        #add noise
        #Cin = np.random.normal(Cin, 0.1*Cin) #10% pump noise

        self.Cins.append(Cin)

        ts = [0, self.sampling_time]

        sol = odeint(self.sdot, self.S, ts, args=(Cin,))[1:]

        self.S = sol[-1,:]

        #print(self.S)
        #self.sSol = np.append(self.sSol, np.random.normal(self.S.reshape(1,len(self.S)), self.S.reshape(1,len(self.S))*0.05), axis = 0)
        self.sSol = np.append(self.sSol,self.S.reshape(1,len(self.S)), axis = 0)
        self.state = self.get_state()

        reward, done = self.reward_func(self.S[0:2], None,None) # use this for custom transition cost

        return self.state, reward, done, None


    def get_state(self):
        '''
        Gets the state (scaled bacterial populations) to be observed by the agent

        Returns:
            scaled bacterial populations
        '''
        Ns = self.S[0:self.num_species]
        return np.array(Ns)/self.scaling

    def action_to_Cin(self,action):
        '''
        Takes a discrete action index and returns the corresponding continuous state
        vector

        Paremeters:
            action: the descrete action
            num_species: the number of bacterial populations
            num_Cin_states: the number of action states the agent can choose from
                for each species
            Cin_bounds: list of the upper and lower bounds of the Cin states that
                can be chosen
        Returns:
            state: the continuous Cin concentrations correspoding to the chosen
                action
        '''

        # calculate which bucket each eaction belongs in
        buckets = np.unravel_index(action, [self.num_Cin_states] * self.num_controlled_species)

        # convert each bucket to a continuous state variable
        Cin = []
        for r in buckets:
            Cin.append(self.Cin_bounds[0] + r*(self.Cin_bounds[1]-self.Cin_bounds[0])/(self.num_Cin_states-1))

        Cin = np.array(Cin).reshape(self.num_controlled_species,)

        return np.clip(Cin, 0, 0.1)

    def reset(self,initial_S = None):
        '''
        Resets env to inital state:

        Parameters:
            initial_S (optional) the initial state to be reset to if different to the default
        Returns:
            The state to be observed by the agent
        '''

        if initial_S is None:
            self.S = np.array(self.initial_S)
        else:
            self.S = np.array(initial_S)

        self.state = self.S[0:self.num_species]

        self.sSol = np.array([self.S])
        initial_Cin = np.array([0.05])
        self.Cins = [initial_Cin]


        return self.get_state()

    def plot_trajectory(self, indices= [0,1]):
        '''
        Creates a matplotlib figure of the time evolution of env variables:

        Parameters:
            indices: the indices of the variables to plot
        '''

        plt.figure()
        xSol = np.array(self.sSol)

        for i in indices:
            plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = self.labels[i])
        plt.ylim(bottom=0)
        plt.legend()

    def save_trajectory(save_path):
        '''
        Saves a numpy array of the time evolution of all enviroment variables

        Parameters:
            save_path: the directory in which to save
        '''
        np.save(save_path + '/final_trajectory', self.sSol)


class ProductEnv():

    '''
    Chemostat environement that can handle an arbitrary number of bacterial strains where all are being controlled and simulates product formation

    '''

    def __init__(self, param_file,  sampling_time, scaling):
        '''
        Parameters:
            param_file: path of a yaml file contining system parameters
            reward_func: python function used to coaculate reward: reward = reward_func(state, action, next_state)
            sampling_time: time between sampl-and-hold intervals
            scaling: population scaling to prevent neural network instability in agent, aim to have pops between 0 and 1. env returns populations/scaling to agent
        '''
        f = open(param_file)
        param_dict = yaml.load(f)
        f.close()
        #self.validate_param_dict(param_dict)
        param_dict = self.convert_to_numpy(param_dict)
        self.set_params(param_dict)
        print(self.initial_N)
        self.initial_S = np.append(np.append(np.append(self.initial_N, self.initial_C), self.initial_C0),self.initial_chems)

        self.S = self.initial_S
        self.sSol = np.array(self.initial_S).reshape(1,len(self.S))
        self.labels = ['N1', 'N2', 'C1', 'C2', 'C0', 'A', 'B', 'P']
        self.Cins = []


        self.sampling_time = sampling_time

        self.scaling = scaling
        self.state = self.get_state()

    def convert_to_numpy(self,param_dict):
        '''
        Takes a parameter dictionary and converts the required parameters into numpy
        arrays

        Parameters:
            param_dict: the parameter dictionary
        Returns:
            param_dict: the converted parameter dictionary
        '''

        # convert all relevant parameters into numpy arrays
        param_dict['ode_params'][2], param_dict['ode_params'][3],param_dict['ode_params'][4], param_dict['ode_params'][5],  param_dict['ode_params'][6],  param_dict['ode_params'][7]  = \
            np.array(param_dict['ode_params'][2]), np.array(param_dict['ode_params'][3]), np.array(param_dict['ode_params'][4]),np.array(param_dict['ode_params'][5]), np.array(param_dict['ode_params'][6]), np.array(param_dict['ode_params'][7])


        param_dict['env_params'][3],param_dict['env_params'][4], param_dict['env_params'][5], param_dict['env_params'][6] = \
             np.array(param_dict['env_params'][3]), np.array(param_dict['env_params'][4]),np.array(param_dict['env_params'][5]), np.array(param_dict['env_params'][6])

        return param_dict

    def validate_param_dict(self,param_dict):
        '''
        Performs input validation on the parameter dictionary supplied by the user
        and throws an error if parameters are invalid.

        Parameters:
            param_dict: the parameter dictionary
        '''

        # validate ode_params
        ode_params = param_dict['ode_params']

        if ode_params[0] <= 0:
            raise ValueError("C0in needs to be positive")
        if ode_params[1] <= 0:
            raise ValueError("q needs to be positive")
        if not all(y > 0 for y in ode_params[2]) or not all(y3 > 0 for y3 in ode_params[3]):
            raise ValueError("all bacterial yield constants need to be positive")
        if not all(Rmax > 0 for Rmax in ode_params[4]):
            raise ValueError("all maximum growth rates need to be positive")
        if not all(Km >= 0 for Km in ode_params[5]) or not all(Km3 >= 0 for Km3 in ode_params[6]):
            raise ValueError("all saturation constants need to be positive")


        # validate Q_params
        env_params = param_dict['env_params']
        num_species = env_params[0]

        if num_species < 0 or not isinstance(num_species, int):
            raise ValueError("num_species needs to be a positive integer")

        if env_params[1] > num_species or env_params[1] < 0 or not isinstance(env_params[1], int):
            raise ValueError("num_controlled_species needs to be a positive integer <= to num_species")

        if env_params[2] < 0 or not isinstance(num_species, int):
            raise ValueError("num_Cin_states needs to be a positive integer")
        if len(env_params[3]) != 2 or env_params[3][0] < 0 or env_params[3][0] >= env_params[3][1]:
            raise ValueError("Cin_bounds needs to be a list with two values in ascending order")


        if not all(x > 0 for x in env_params[4]):
            raise ValueError("all initial populations need to be positive")
        if not all(c > 0 for c in env_params[5]):
            raise ValueError("all initial concentrations need to be positive")
        if env_params[6] < 0:
            raise ValueError("initial C0 needs to be positive")

    def set_params(self,param_dict):
        '''
        Sets env params to those stored in a python dictionary
            Parameters:
                param_dict : ptyhon dictionary containing all params
        '''
        self.C0in, self.q, self.y, self.y0, self.umax, self.Km, self.Km0, self.A = param_dict['ode_params']
        self.num_species, self.num_controlled_species, self.num_Cin_states, self.Cin_bounds, self.initial_N, self.initial_C, self.initial_C0, self.initial_chems = param_dict['env_params']

    def monod(self,C, C0):
        '''
        Calculates the growth rate based on the monod equation

        Parameters:
            C: the concetrations of the auxotrophic nutrients for each bacterial
                population
            C0: concentration of the common carbon source
            Rmax: array of the maximum growth rates for each bacteria
            Km: array of the saturation constants for each auxotrophic nutrient
            Km0: array of the saturation constant for the common carbon source for
                each bacterial species
        '''

        # convert to numpy

        growth_rate = ((self.umax*C)/ (self.Km + C)) * (C0/ (self.Km0 + C0))

        return growth_rate

    def sdot(self,S, t, Cin):
        '''
        Calculates and returns derivatives for the numerical solver odeint

        Parameters:
            S: current state
            t: current time
            Cin: array of the concentrations of the auxotrophic nutrients and the
                common carbon source
            params: list parameters for all the exquations
            num_species: the number of bacterial populations
        Returns:
            dsol: array of the derivatives for all state variables
        '''

        # extract variables
        N = np.array(S[:self.num_species])
        C = np.array(S[self.num_species:self.num_species+self.num_controlled_species])
        C0 = np.array(S[4])
        A = np.array(S[5])
        B = np.array(S[6])
        P = np.array(S[7])

        R = self.monod(C, C0)

        Cin = Cin[:self.num_controlled_species]

        # calculate derivatives
        dN = N * (R.astype(float) + np.matmul(self.A, N) - self.q) # q term takes account of the dilution
        dC = self.q*(Cin - C) - (1/self.y)*R*N # sometimes dC.shape is (2,2)
        dC0 = self.q*(self.C0in - C0) - sum(1/self.y0[i]*R[i]*N[i] for i in range(self.num_species))
        dA = N[0] - 2*A**2*B - self.q*A
        dB = N[1] - A**2*B - self.q*B
        dP = A**2*B - self.q*P
        if dC.shape == (2,2):
            print(q,Cin.shape,C0,C,y,R,N) #C0in

        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])

        dsol = np.append(dN, dC)
        dsol = np.append(dsol, dC0)
        dsol = np.append(dsol, dA)
        dsol = np.append(dsol, dB)
        dsol = np.append(dsol, dP)


        return dsol

    def step(self, action):
        '''
        Performs one sampling and hold interval using the action provided by a reinforcment leraning agent

        Parameters:
            action: action chosen by agent
        Returns:
            state: scaled state to be observed by agent
            reward: reward obtained buring this sample-and-hold interval
            done: boolean value indicating whether the environment has reached a terminal state


        '''
        Cin = self.action_to_Cin(action)


        self.Cins.append(Cin)

        ts = [0, self.sampling_time]

        sol = odeint(self.sdot, self.S, ts, args=(Cin,))[1:]

        self.S = sol[-1,:]

        self.sSol = np.append(self.sSol,self.S.reshape(1,len(self.S)), axis = 0)
        self.state = self.get_state()

        reward, done = self.reward_func(self.state, None,None) # use this for custom transition cost

        return self.state, reward, done, None

    def get_state(self):
        '''
        Gets the state (scaled bacterial populations) to be observed by the agent

        Returns:
            scaled bacterial populations
        '''
        Ns = self.S[0:self.num_species]
        return np.array(Ns)/self.scaling


    def action_to_Cin(self,action):
        '''
        Takes a discrete action index and returns the corresponding continuous state
        vector

        Paremeters:
            action: the descrete action
            num_species: the number of bacterial populations
            num_Cin_states: the number of action states the agent can choose from
                for each species
            Cin_bounds: list of the upper and lower bounds of the Cin states that
                can be chosen
        Returns:
            state: the continuous Cin concentrations correspoding to the chosen
                action
        '''

        # calculate which bucket each eaction belongs in
        buckets = np.unravel_index(action, [self.num_Cin_states] * self.num_controlled_species)

        # convert each bucket to a continuous state variable
        Cin = []
        for r in buckets:
            Cin.append(self.Cin_bounds[0] + r*(self.Cin_bounds[1]-self.Cin_bounds[0])/(self.num_Cin_states-1))

        Cin = np.array(Cin).reshape(self.num_controlled_species,)
        
        return np.clip(Cin, 0, 0.1)

    def reset(self,initial_S = None):
        '''
        Resets env to inital state:

        Parameters:
            initial_S (optional) the initial state to be reset to if different to the default
        Returns:
            The state to be observed by the agent
        '''
        if initial_S is None:
            self.S = np.array(self.initial_S)
        else:
            self.S = np.array(initial_S)

        self.state = self.S[0:self.num_species]

        self.sSol = np.array([self.S])
        initial_Cin = np.array([0.05])
        self.Cins = [initial_Cin]


        return self.get_state()

    def plot_trajectory(self, indices= [0,1]):
        '''
        Creates a matplotlib figure of the time evolution of env variables:

        Parameters:
            indices: the indices of the variables to plot
        '''

        plt.figure()
        xSol = np.array(self.sSol)

        for i in indices:
            plt.plot(np.linspace(0, len(xSol[:,0]) ,len(xSol[:,0])), xSol[:,i], label = self.labels[i])
        plt.ylim(bottom=0)
        plt.legend()

    def save_trajectory(save_path):
        '''
        Saves a numpy array of the time evolution of all enviroment variables

        Parameters:
            save_path: the directory in which to save
        '''
        np.save(save_path + '/final_trajectory', self.sSol)

    def reward_func(self, state, action, next_state):
        '''
        Reward function for the optimisation of product formation

        '''
        return self.q*self.S[-1]/100000, False
