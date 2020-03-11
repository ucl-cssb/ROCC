
import yaml
import numpy as np
from scipy.integrate import odeint


import matplotlib.pyplot as plt


class SingleAuxotrophChemostatEnv(ChemostatEnv):

    '''
    class for the system where we have two strains, only one of which is an auxotroph and is controlled
    '''

    def __init__(self, param_file, update_timesteps):
        ChemostatEnv.__init__(self,param_file, update_timesteps)
        self.labels = ['N1', 'N2', 'C', 'C0']


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

        u1 = self.umax[0]*(C0/ (self.Km0[0] + C0));
        u2 = (self.umax[1]*C/ (self.Km + C)) * (C0/ (self.Km0[1] + C0));

        return np.array([u1, u2])

    def sdot(self,S, t, Cin):  #YES
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
        dC = self.q*(Cin - C) - (1/self.y)*R[1]*N[1]# sometimes dC.shape is (2,2)
        dC0 = self.q*(self.C0in - C0) - sum(1/self.y0[i]*R[i]*N[i] for i in range(self.num_species))




        if dC.shape == (2,2):
            print(q,Cin.shape,C0,C,y,R,N) #C0in

        # consstruct derivative vector for odeint
        dC0 = np.array([dC0])
        dsol = np.append(dN, dC)
        dsol = np.append(dsol, dC0)


        return dsol

class SimpleChemostatEnv(ChemostatEnv):

    '''
    class for the simplest chemostat system, where we have only one strain and we control the carbon source in
    '''

    def __init__(self, param_file, sampling_time, update_timesteps):
        self.update_timesteps = update_timesteps

        f = open(param_file)
        param_dict = yaml.load(f)
        f.close()
        #self.validate_param_dict(param_dict)
        param_dict = self.convert_to_numpy(param_dict)
        self.set_params(param_dict)

        self.initial_S = np.append(self.initial_N,self.initial_C0)
        self.S = self.initial_S

        self.sSol = np.array(self.initial_S).reshape(1,len(self.S))
        self.labels = ['N', 'C0']
        self.Cins = [self.initial_C0]
        self.state = self.get_state()
        self.sampling_time = sampling_time


    def convert_to_numpy(self, param_dict):
        '''
        Takes a parameter dictionary and converts the required parameters into numpy
        arrays

        Parameters:
            param_dict: the parameter dictionary
        Returns:
            param_dict: the converted parameter dictionary
        '''

        # convert all relevant parameters into numpy arrays
        param_dict['ode_params'][1], param_dict['ode_params'][2],param_dict['ode_params'][3] = \
            np.array(param_dict['ode_params'][1]), np.array(param_dict['ode_params'][2]), np.array(param_dict['ode_params'][3])


        param_dict['env_params'][2],param_dict['env_params'][3], param_dict['env_params'][4] = \
             np.array(param_dict['env_params'][2]), np.array(param_dict['env_params'][3]),np.array(param_dict['env_params'][4])

        return param_dict #YES

    def validate_param_dict(self, param_dict):
        '''
        Performs input validation on the parameter dictionary supplied by the user
        and throws an error if parameters are invalid.

        Parameters:
            param_dict: the parameter dictionary
        '''

        # validate ode_params
        ode_params = param_dict['ode_params']

        if ode_params[0] <= 0:
            raise ValueError("q needs to be positive")
        if not all(y > 0 for y in ode_params[1]) or not all(y3 > 0 for y3 in ode_params[2]):
            raise ValueError("all bacterial yield constants need to be positive")
        if not all(Rmax > 0 for Rmax in ode_params[3]):
            raise ValueError("all maximum growth rates need to be positive")
        if not all(Km >= 0 for Km in ode_params[4]):
            raise ValueError("all saturation constants need to be positive")


        # validate Q_params
        env_params = param_dict['env_params']
        num_species = env_params[0]

        if num_species < 0 or not isinstance(num_species, int):
            raise ValueError("num_species needs to be a positive integer")

        if env_params[1] < 0 or not isinstance(num_species, int):
            raise ValueError("num_Cin_states needs to be a positive integer")

        if len(env_params[2]) != 2 or env_params[2][0] < 0 or env_params[2][0] >= env_params[2][1]:
            raise ValueError("Cin_bounds needs to be a list with two values in ascending order")


        if not all(x > 0 for x in env_params[3]):
            raise ValueError("all initial populations need to be positive")
        if not all(c > 0 for c in env_params[4]):
            raise ValueError("all initial concentrations need to be positive")

    def set_params(self,param_dict): #YES
        self.q, self.y0, self.umax, self.Km0 = param_dict['ode_params']
        self.num_species, self.num_Cin_states, self.Cin_bounds, self.initial_N, self.initial_C0 = param_dict['env_params']
        self.num_controlled_species = 1

    def monod(self,C0):
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

        growth_rate = self.umax * (C0/ (self.Km0 + C0))
        return growth_rate #YES

    def sdot(self,S, t, Cin):  #YES
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
        N = np.array(S[0])
        C0 = np.array(S[1])

        R = self.monod(C0)

        C0in = Cin[0]

        # calculate derivatives
        dN = N * (R.astype(float) - self.q) # q term takes account of the dilution
        dC0 = self.q*(C0in - C0) - 1/self.y0*R*N


        # consstruct derivative vector for odeint
        dsol = np.append(dN, dC0)

        return dsol
