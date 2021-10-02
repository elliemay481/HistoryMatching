import numpy as np
import matplotlib.pyplot as plt

#import random
#import matplotlib.gridspec as gridspec
#from scipy.stats import norm, uniform
#from scipy import stats
#import nestle

import emulators
import plot
import utils
import data

class Results(dict):
    """
    For accessing the outputs of the history match.
    """

    def __getattribute__(self, name):
        """ Returns attribute of history matching results."""
        attr = self[name]
        if not attr:
           raise AttributeError(f'{self.__class__.__name__}.{name} does not exist.')
        return attr




class HistoryMatch:

    """

    Args
    ----
    obs_data : array
        1-d array of observational data. Each element of the array corresponds to a model output.
    
    ndim : int
        Number of unknown parameters.
    
    model : list of functions
        List of functions representing the true model of the system.
        The input of the functions are the ndim unknown parameters.

    emulator : {'GP', 'EC'}
        Type of emulator used to emulate the model. Choices are Gaussian process
        ('GP') and eigenvector continuation ('EC').

    volume_shape : {'hypercube', 'ellipsoid'}
        Determines the final shape of the nonimplausible parameter volume.

    bounds : array of floats
        (ndim x 2) array containing the upper and lower bounds for each parameter.
    
    nwaves : int, optional
        Number of history matching waves to complete. If None (default is None)
        scheme will continue until termination conditions are met. Number
        of waves may be less than nwaves if conditions reached earlier.

    ntraining : int, optional

    nsamples : int, optional

    """
    
    def __init__(self, obs_data, ndim, model, emulator, volume_shape, bounds,
                    var_obs, var_method, ntraining=None, nsamples=None):

        self.Z = obs_data
        self.ndim = ndim
        self.bounds = bounds
        self.model = model
        self.noutputs = len(obs_data)
        self.var_obs = var_obs
        self.var_method = var_method

        self.emulator_choice = emulator

        if ntraining:
            self.ntraining = ntraining
        else:
            self.ntraining = 20*ndim

        if nsamples:
            self.nsamples = nsamples
        else:
            self.nsamples = 5000*ndim

        self.shape = volume_shape

    def implausibility(self, E, z_i, var_em, var_method, var_obs):

        """
        Evaluates the implausibility measure given emulator output and observational
        data.

        Args
        ----
        E : array
            Expectation values given as output from the emulator

        z_i : float
            Observational datapoint

        var_em : float
            Variance of emulator uncertainty

        var_method : float
            Variance of method uncertainty

        var_obs : float
            Variance of observational uncertainty

        va

        """
        
        # E - emulator expectation
        # z - observational data
        # var_em - emulator_uncertainty
        # var_md - model discrepency error
        # var_obs - observational error
    
        return np.sqrt( ( E - z_i )**2  /  ( var_em + var_method + var_obs ) )


    def wave(self, bounds):

        theta_train, theta_test = data.prepare_data(self.ndim, self.nsamples, self.ntraining, self.bounds)

        implausibilities_all = np.zeros((self.nsamples, self.noutputs))

        for output in range(self.noutputs):

            Ztrain = self.model[output](*theta_train.T)

            if self.emulator_choice == 'GP':
                GP = emulators.Gaussian_Process(theta_train, theta_test, Ztrain)
            elif self.emulator_choice == 'EC':
                print('EC not yet developed')

            
            GP.optimize()
            mu, cov, sd = GP.emulate()


            for i in range(len(theta_test)):
                implausibilities_all[i, output] = self.implausibility(mu[i], self.Z[output], sd[i], self.var_method, self.var_obs)
        
    
        # get index of second highest maximum implaus for all outputs
        max_I = implausibilities_all.argsort()[:,-2]
        implausibilities = implausibilities_all[range(len(max_I)), max_I]

        samples = np.concatenate((theta_test, implausibilities.reshape(-1,1)), axis=1)
        nonimplausible = np.delete(samples, np.where(samples[:,-1] > 3), axis=0)

        if self.shape == 'hypercube':
            nonimplausible_bounds = locate_boundaries(nonimplausible, ndim)
        else:
            print('elliptical not developed yet')

        return nonimplausible_bounds, nonimplausible, samples



    def run(self, nwaves):

        """
        Performs waves of history matching to find nonimplausible parameter space.

        Returns
        -------
        result: 'Result'
            Dictionary containing results of each wave

        """

        if nwaves:
            self.nwaves = nwaves
        else:
            self.nwaves = 1

        # run number of waves. in each wave:
            # for each output in obs_data, emulate and evaluate implausibility
            # then combine to determine max implausibility
            # either find ellipses or cutoffs
            # construct new nonimplaus region
            # save results of wave

        # initialise training set and parameter space
        #theta_train, theta_test = data.prepare_data(self.ndim, self.nsamples, self.ntraining, self.bounds)
        nonimplausible_bounds = self.bounds

        # calculate initial parameter volume
        initial_volume = utils.hypercube_volume(self.ndim, self.bounds)
        
        bounds_list = []
        nonimp_region_list = []
        sample_list = []

        nonimplausible_bounds = self.bounds

        for wave in range(self.nwaves):
            print('Running wave ' + str(wave+1))
            nonimplausible_bounds, nonimplausible_region, samples = self.wave(nonimplausible_bounds)
            
            bounds_list.append(nonimplausible_bounds)
            nonimp_region_list.append(nonimplausible_region)
            sample_list.append(samples)

            nonimplausible_volume = utils.hypercube_volume(self.ndim, nonimplausible_bounds)
            print('Relative nonimplausible volume remaining: ' + str(round(nonimplausible_volume/initial_volume,3)))

        return Results({'bounds': bounds_list, 'regions': nonimp_region_list, 'samples': sample_list})

            

            

            
                