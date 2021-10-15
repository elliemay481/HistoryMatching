# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
import emulators
import plot
import utils
import sample

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
    
    simulator : function

        Simulator that generates outputs given an input of parameter values.
        Output should be an array of the same size as obs_data. Must be defined
        with the Simulator class.

    emulator : {'GP', 'EC'}
        Type of emulator used to emulate the model. Choices are Gaussian process
        ('GP') and eigenvector continuation ('EC').

    volume_shape : {'hypercube', 'ellipsoid'}
        Determines the final shape of the nonimplausible parameter volume.
    
    nwaves : int, optional
        Number of history matching waves to complete. If None (default is None)
        scheme will continue until termination conditions are met. Number
        of waves may be less than nwaves if conditions reached earlier.

    ntraining : int, optional

    nsamples : int, optional

    """
    
    def __init__(self, obs_data, ndim, emulator='GP', volume_shape='hypercube', ntraining=None, nsamples=None):

        self.ndim = ndim

        self.simulator = None
        
        self.emulator_choice = emulator

        if ntraining:
            self.ntraining = ntraining
        else:
            self.ntraining = 20

        if nsamples:
            self.nsamples = nsamples
        else:
            self.nsamples = 5000

        self.shape = volume_shape

    
    def save_model(self, output_filename):
        None

    def initialize_volume(self, min, max):

        '''
        Args
        ---
        min : ndarray, shape (ndim,)
        Array of the minimum parameter values to consider.

        max : ndarray, shape (ndim,)
        Array of the maximum parameter values to consider.
        
        '''

        self.nonimplausible_volume = self.hypercube(min, max)

    def set_observations(self, obs_data, sigma_obs=0, sigma_method=0,
                            sigma_model=0, sigma_other = 0):
        sigmas = [sigma_obs, sigma_method, sigma_model, sigma_other]
        #assert not all(s == 0 for s in sigmas), "At least one standard deviation must be nonzero."
        self.Z = obs_data
        self.sigma_obs = sigma_obs
        self.var_obs = sigma_obs**2
        self.var_method = sigma_method**2
        self.var_model = sigma_model**2
        self.var_other = sigma_other**2
        self.noutputs = len(obs_data)


    def hypercube(self, min, max):
        return np.vstack((np.array(min), np.array(max))).T

    def ellipsoid(min,max):
        None



    def simulate(self, theta):
        if self.simulator is None:
            raise NotImplementedError("Simulator not defined")
        else:
            return self.simulator(theta)
            # **** check output here ******

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

        var_model : float
            Variance of model uncertainty

        var_obs : float
            Variance of observational uncertainty

        va

        """
        
    
        return np.sqrt( ( E - z_i )**2  /  ( var_em + var_method + var_obs ) )


    def wave(self, nonimplausible_volume, theta_train, theta_test, n_active_params):

        implausibilities_all = np.zeros((self.nsamples, self.noutputs))

        # run simulator for training points
        Ztrain = self.simulator(*theta_train.T)

        Ztest = self.simulator(*theta_test.T)



        for output in range(self.noutputs):
        #for output in range(n_active_params):

            Ztrain_i = Ztrain[output]
            #print(Ztrain_i)

            if self.emulator_choice == 'GP':
                GP = emulators.GaussianProcess(theta_train, Ztrain_i, length_scale=2, signal_sd=0.1, ols_order=1, bayes_linear = True, noise_sd = 0.01)
            elif self.emulator_choice == 'EC':
                print('EC not yet developed')

            print('Emulating...')
            #mu0, cov0, sd0 = GP.emulate(theta_test)
            #GP.optimize()
            mu, sd = GP.emulate(theta_test)

            #mu = Ztest[output]

            #sd = np.ones_like(mu) * 0.01
            #print(np.mean(sd))
            #print(self.sigma_obs)

            if np.mean(sd) < np.sqrt(self.var_obs[output]):
                print('Mean emulator s.d lower than obs')

            
            for i in range(len(theta_test)):
                implausibilities_all[i, output] = self.implausibility(mu[i], self.Z[output], sd[i]**2, self.var_method, self.var_obs[output])
        
    
        # get index of second highest maximum implaus for all outputs
        max_I = implausibilities_all.argsort()[:,-2]
        implausibilities = implausibilities_all[range(len(max_I)), max_I]

        samples = np.concatenate((theta_test, implausibilities.reshape(-1,1)), axis=1)
        nonimplausible = np.delete(samples, np.where(samples[:,-1] > 3), axis=0)

        if self.shape == 'hypercube':
            self.nonimplausible_volume = utils.locate_boundaries(nonimplausible, self.ndim)
        else:
            # ****** fix **********
            self.nonimplausible_volume = utils.locate_boundaries(nonimplausible, self.ndim)

        return self.nonimplausible_volume, nonimplausible, samples, mu, sd, Ztrain[-1]



    def run(self, nwaves):

        """
        Performs waves of history matching to find nonimplausible parameter space.

        Returns
        -------
        result: 'Result'
            Dictionary containing results of each wave

        """

        if self.simulator is None:
            raise NotImplementedError("Observational data not initialised")

        if nwaves:
            self.nwaves = nwaves
        else:
            self.nwaves = 1

        # run number of waves. in each wave:

        # initialise training set and parameter space
        theta_train, theta_test = sample.hypercube_sample(self.ndim, self.nsamples, self.ntraining, self.nonimplausible_volume)

        # calculate initial parameter volume
        initial_volume = utils.hypercube_volume(self.ndim, self.nonimplausible_volume)
        
        nonimp_volumes = []
        nonimp_region_list = []
        sample_list = []
        test_list = []
        train_list = []
        em_mu = []
        em_sd = []
        z_train_list = []

        n_active_params = 16

        for wave in range(self.nwaves):
            print('Running wave ' + str(wave+1))

            test_list.append(theta_test)
            train_list.append(theta_train)
                
            # run history matching wave
            self.nonimplausible_volume, nonimplausible_region, samples, mu, sd, z_train = self.wave(self.nonimplausible_volume, theta_train, theta_test, n_active_params)

            em_mu.append(mu)
            em_sd.append(sd)
            z_train_list.append(z_train)

            n_active_params =+ 2
            
            
            # generate well space samples in parameter space
            if self.shape == 'hypercube':
                theta_train, theta_test = sample.hypercube_sample(self.ndim, self.nsamples, self.ntraining, self.nonimplausible_volume)

            elif self.shape == 'ellipsoid':
                
                # find mean and covariance of samples
                K0 = np.cov(nonimplausible_region[:,:-1].T)

                mean = np.mean(nonimplausible_region[:,:-1], axis=0)
                theta_train, theta_test = sample.ellipsoid_sample(self.ndim, self.nsamples, self.ntraining, mean, K0)

                # discard sample points outside of boundaries
                # **** FIX BOUNDARIES HERE ********
                
                theta_test_reduced = np.delete(theta_test, np.where(np.abs(theta_test) > 1), axis=0)
                theta_train_reduced = np.delete(theta_train, np.where(np.abs(theta_train) > 1), axis=0)
            
                '''
                while len(theta_test_reduced) < self.nsamples:
                    N_te = self.nsamples - len(theta_test_reduced)
                    theta_train_new, theta_test_new = sample.ellipsoid_sample(self.ndim, N_te, 0, mean, K0)
                    theta_test_new = np.delete(theta_test_new, np.where(np.abs(theta_test_new) > 1)[0], axis=0)
                    theta_test_reduced = np.concatenate((theta_test_reduced, theta_test_new), axis=0)

                while len(theta_train_reduced) < self.ntraining:
                    N_tr = self.ntraining - len(theta_train_reduced)
                    theta_train_new, theta_test_new = sample.ellipsoid_sample(self.ndim, 0, N_tr, mean, K0)
                    theta_train_new = np.delete(theta_train_new, np.where(np.abs(theta_train_new) > 1)[0], axis=0)
                    theta_train_reduced = np.concatenate((theta_train_reduced, theta_train_new), axis=0)

                theta_test = theta_test_reduced
                theta_train = theta_train_reduced

                def Mahalanobis(sample, mean, covariance):
                    M = np.sqrt((sample - mean).T.dot(np.linalg.inv(covariance).dot((sample - mean))))
                    return M'''
                '''
                delete_pt = 0
                test_mean = np.mean(theta_test, axis=0)
                test_cov = np.cov(theta_test.T)
                for i in range(len(theta_test)):
                    M = Mahalanobis(theta_test[i], test_mean, test_cov)
                    if M > 3:
                        np.delete(theta_test, i)
                        delete_pt += 1

            print(delete_pt)
            
            
            '''

            nonimp_volumes.append(self.nonimplausible_volume)
            nonimp_region_list.append(nonimplausible_region)
            sample_list.append(samples)

            cube_volume = utils.hypercube_volume(self.ndim, self.nonimplausible_volume)
            print('Relative nonimplausible volume remaining: ' + str(round(cube_volume/initial_volume,3)))

        return Results({'nonimp_volumes': nonimp_volumes, 'regions': nonimp_region_list, 'samples': sample_list, 'train_pts': train_list, 'test_pts': test_list, 'emulator_mu': em_mu, 'emulator_sd': em_sd, 'z_train': z_train_list})




class Simulator(HistoryMatch): 
    def __init__(self, hm_instance):
        self.hm_instance = hm_instance

    def set_simulator(self, model):
        self.hm_instance.simulator = model

            

            
                