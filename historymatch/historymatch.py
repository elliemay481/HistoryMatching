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
            self.ntraining = 8

        if nsamples:
            self.nsamples = nsamples
        else:
            self.nsamples = 8000

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


    def wave(self, nonimplausible_volume, theta_train, theta_test):

        implausibilities_all = np.zeros((self.nsamples, self.noutputs))

        # for implausibility cutoff determination
        implausibilities_data = np.zeros((self.nsamples, self.noutputs))
        Ztest = self.simulator(*theta_test.T)

        # run simulator for training points
        Ztrain = self.simulator(*theta_train.T)

        # for emulator plots
        theta1 = np.linspace(-1,1,100)
        theta2 = 0.3*np.ones(100)
        theta3 = 0.4*np.ones(100)
        #test_grid = np.concatenate((theta1.reshape(-1,1), theta2.reshape(-1,1), theta3.reshape(-1,1)), axis=1)
        #test_grid = np.concatenate((theta1.reshape(-1,1), theta2.reshape(-1,1), theta3.reshape(-1,1)), axis=1)

        #Z_grid = self.simulator(*test_grid.T)
        Z_grid = self.simulator(*theta_test.T)

        output_convergence = np.full(self.noutputs, False, dtype=bool)


        for output in range(self.noutputs):

            Ztrain_i = Ztrain[output]

            if self.emulator_choice == 'GP':
                GP = emulators.GaussianProcess(theta_train, Ztrain_i, length_scale=2, signal_sd=0.1, ols_order=1, bayes_linear = True, noise_sd = 1e-9)
            elif self.emulator_choice == 'EC':
                print('EC not yet developed')

            print('Emulating output {}...'.format(output))

            
            #GP.optimize()
            #mu_grid, sd_grid = GP.emulate(test_grid)

            mu, sd = GP.emulate(theta_test)

            

            #mu = Ztest[output]

            #sd = np.ones_like(mu) * 0.1

            mu_grid = np.zeros_like(mu)
            sd_grid = np.zeros_like(mu)

            if np.mean(sd) + 3*np.sqrt(np.var(sd)) < self.sigma_obs[output]:
                output_convergence[output] = True
            else:
                output_convergence[output] = False

            
            for i in range(len(theta_test)):
                implausibilities_all[i, output] = self.implausibility(mu[i], self.Z[output], sd[i]**2, self.var_method, self.var_obs[output])


            # for implausibility cutoff determination
            for i in range(len(theta_test)):
                implausibilities_data[i, output] = self.implausibility(Ztest[output][i], self.Z[output], 0, self.var_method, self.var_obs[output])
        

        # get index of second highest maximum implaus for all outputs
        max_I = implausibilities_all.argsort()[:,-1]
        max2_I = implausibilities_all.argsort()[:,-2]
        implausibilities = implausibilities_all[range(len(max2_I)), max2_I]

        # for implausibility cutoff determination
        max_Idata = implausibilities_data.argsort()[:,-1]
        Idata = implausibilities_data[range(len(max_Idata)), max_Idata]

        samples = np.concatenate((theta_test, implausibilities.reshape(-1,1)), axis=1)
        print(samples.shape)
        nonimplausible = np.delete(samples, np.where(samples[:,-1] > 2.7), axis=0)
        print(nonimplausible.shape)

        if self.shape == 'hypercube':
            self.nonimplausible_volume = utils.locate_boundaries(nonimplausible, self.ndim)
        else:
            # ****** fix **********
            self.nonimplausible_volume = utils.locate_boundaries(nonimplausible, self.ndim)

        return self.nonimplausible_volume, nonimplausible, samples, mu, sd, Z_grid[-1], mu_grid, sd_grid, output_convergence, implausibilities, Idata



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
            self.nwaves = 10

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
        z_grid_list = []
        Itrain_list = []
        Idata_list = []

        for wave in range(self.nwaves):
            print('Running wave ' + str(wave+1))

            test_list.append(theta_test)
            train_list.append(theta_train)
                
            # run history matching wave
            self.nonimplausible_volume, nonimplausible_region, samples, mu, sd, z_grid, mu_test, sd_test, output_convergence, Itrain, Idata \
                                                = self.wave(self.nonimplausible_volume, theta_train, theta_test)

            print('Convergence : ' + str(np.all(output_convergence)))

            #if np.all(output_convergence) = True:
                #end_str = 'Convergence reached after ' + str(wave) + ' waves.'
                #return end_str

            em_mu.append(mu_test)
            em_sd.append(sd_test)
            z_grid_list.append(z_grid)
            Itrain_list.append(Itrain)
            Idata_list.append(Idata)

            
            
            # generate well space samples in parameter space
            if self.shape == 'hypercube':
                theta_train, theta_test = sample.hypercube_sample(self.ndim, self.nsamples, self.ntraining, self.nonimplausible_volume)

            elif self.shape == 'ellipsoid':
                
                # find mean and covariance of samples
                K0 = np.cov(nonimplausible_region[:,:-1].T)

                mean = np.mean(nonimplausible_region[:,:-1], axis=0)
                theta_train, theta_test = sample.ellipsoid_sample(self.ndim, self.nsamples, self.ntraining, mean, K0)


            nonimp_volumes.append(self.nonimplausible_volume)
            nonimp_region_list.append(nonimplausible_region)
            sample_list.append(samples)

            cube_volume = utils.hypercube_volume(self.ndim, self.nonimplausible_volume)
            print('Relative nonimplausible volume remaining: ' + str(round(cube_volume/initial_volume,3)))

        return Results({'nonimp_volumes': nonimp_volumes, 'regions': nonimp_region_list, 'samples': sample_list, 'train_pts': train_list, 'test_pts': test_list, 'emulator_mu': em_mu, 'emulator_sd': em_sd, 'z_grid': z_grid_list, 'Itrain': Itrain_list, 'Idata': Idata_list})




class Simulator(HistoryMatch): 
    def __init__(self, hm_instance):
        self.hm_instance = hm_instance

    def set_simulator(self, model):
        self.hm_instance.simulator = model

            

            
                