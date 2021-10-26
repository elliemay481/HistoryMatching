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
    
    ndim : int
        Number of unknown parameters.

    emulator : {'GP', 'EC'}
        Type of emulator used to emulate the model. Choices are Gaussian process
        ('GP') and eigenvector continuation ('EC').

    volume_shape : {'hypercube', 'ellipsoid', 'hypercube_rot'}
        Determines the shape of the nonimplausible parameter volume.
    
    nwaves : int, optional
        Number of history matching waves to complete. If None (default is None)
        scheme will continue for 20 waves or until termination conditions are met,
        whichever is sooner. Total number of waves may be less than nwaves if
        termination conditions reached earlier.

    ntraining : int, optional

    nsamples : int, optional

    """
    
    def __init__(self, ndim, emulator='GP', volume_shape='hypercube', ntraining=None, nsamples=None):

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
        self.Z = None

    
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
                            sigma_model=0, sigma_other=0):

        '''
        Args
        ---

        obs_data : array
        1-d array of observational data. Each element of the array corresponds to a model output.

        sigma_method : float
            Standard deviation of method uncertainty

        sigma_model : float
            Standard deviation of model uncertainty

        sigma_obs : ndarray, shape (len(obs_data),)
            Standard deviation of the observational uncertainty for each observational datapoint provided.

        sigma_other : float
            Standard deviation of other uncertainties
        
        '''

        self.sigmas = np.array([sigma_obs, sigma_method, sigma_model, sigma_other])
        assert not all(s == 0 for s in self.sigmas), "At least one standard deviation must be nonzero."
        self.Z = obs_data
        self.sigma_obs = sigma_obs
        self.sigma_method = sigma_method
        self.sigma_model = sigma_model
        self.sigma_other = sigma_other
        self.noutputs = len(obs_data)


    def simulate(self, theta):
        if self.simulator is None:
            raise NotImplementedError("Simulator not defined.")
        else:
            return self.simulator(theta)

    def implausibility(self, E, z_i, var_em, var_method, var_obs, var_model):

        """
        Evaluates the implausibility measure given emulator output and observational
        data.

        Args
        ----
        E : array
            Expectation values given as output from the emulator

        z_i : float
            Observational datapoint

        var :
            Array of uncertainty variances

        """
        
    
        return np.sqrt( ( E - z_i )**2  /  np.sum(var) )


    def wave(self, theta_train, theta_samples):

        """
        Performs a single wave of history matching.

        Args
        ----
        theta_train : ndarray, shape (ntraining, ndim)
            Input parameters used within the simulator.

        theta_samples : ndarray, shape (nsamples, ndim)
            Input parameters used within the emulator.

        Returns
        -------
        self.nonimplausible_volume, nonimplausible_samples, I_samples, mu, sd, output_convergence

        theta_train : ndarray, shape (ntraining, ndim)
            Input parameters used within the simulator.


        """

        # evaluate simulator over training points
        Ztrain = self.simulate(*theta_train.T)

        output_convergence = np.full(self.noutputs, False, dtype=bool)
        implausibilities_all = np.zeros((self.nsamples, self.noutputs))

        for output in range(self.noutputs):

            Ztrain_i = Ztrain[output]

            if self.emulator_choice == 'GP':
                GP = emulators.GaussianProcess(theta_train, Ztrain_i, length_scale=2, signal_sd=0.1, bayes_linear = True, noise_sd = 1e-9)
            elif self.emulator_choice == 'EC':
                print('EC not yet developed')

            # ***** progress bar?? *********
            print('Emulating output {}...'.format(output))

            mu, sd = GP.emulate(theta_samples)

            if np.mean(sd) + 3*np.sqrt(np.var(sd)) < self.sigma_obs[output]:
                output_convergence[output] = True
            else:
                output_convergence[output] = False

            for i in range(len(theta_samples)):
                implausibilities_all[i, output] = self.implausibility(mu[i], self.Z[output], np.square(self.sigmas))
        

        # get index of second highest maximum implaus for all outputs
        max_I = implausibilities_all.argsort()[:,-1]
        max2_I = implausibilities_all.argsort()[:,-2]
        max_implausibilities = implausibilities_all[range(len(max2_I)), max2_I]

        I_samples = np.concatenate((theta_samples, max_implausibilities.reshape(-1,1)), axis=1)
        nonimplausible_samples = np.delete(I_samples, np.where(samples[:,-1] > 3), axis=0)

        if self.shape == 'hypercube':
            self.nonimplausible_bounds = utils.locate_boundaries(nonimplausible, self.ndim)
        elif self.shape == 'hypercube_rot':
            # ***** fix **********
            training_rot, samples_rot = sample.rotated_hypercube_samples(self.ndim, nonimplausible[:,:-1], self.nsamples, self.ntraining)
            self.nonimplausible_bounds = samples_rot
        else:
            # ****** fix **********
            self.nonimplausible_bounds = utils.locate_boundaries(nonimplausible, self.ndim)

        return self.nonimplausible_bounds, nonimplausible_samples, I_samples, mu, sd, output_convergence



    def run(self, nwaves):

        """
        Performs waves of history matching to find nonimplausible parameter space.

        Returns
        -------
        result: 'Result'
            Dictionary containing results of each wave

        """

        if self.Z is None:
            raise NotImplementedError("Observational data not initialised.")

        if nwaves:
            self.nwaves = nwaves
        else:
            self.nwaves = 20

        # initialise training set and parameter space
        theta_train, theta_samples = sample.hypercube_sample(self.ndim, self.nsamples, self.ntraining, self.nonimplausible_bounds)

        # calculate initial parameter volume
        initial_volume = utils.hypercube_volume(self.ndim, self.nonimplausible_bounds)
        
        # initialise results
        nonimp_bounds = []
        nonimp_samples = []
        sample_pts = []
        training_pts = []
        I_samples_list = []
        emulator_output = np.empty((0,2))

        for wave in range(self.nwaves):
            print('Running wave ' + str(wave+1))

            sample_pts.append(theta_samples)
            training_pts.append(theta_train)
                
            # run history matching wave
            self.nonimplausible_bounds, nonimplausible_samples, I_samples, mu, sd, output_convergence,\
                                        = self.wave(theta_train, theta_samples)

            print('Convergence : ' + str(np.all(output_convergence)))

            #if np.all(output_convergence) = True:
                #end_str_conv = 'Convergence reached after ' + str(wave) + ' waves.'
                #return end_str_conv
            
            if nonimplausible_samples.shape[0] == 0:
                end_str = 'Nonimplausible volume empty. Terminated after ' + str(wave) + ' waves.'
                return end_str_empty

            # generate well space samples in parameter space
            if self.shape == 'hypercube':
                theta_train, theta_samples = sample.hypercube_sample(self.ndim, self.nsamples, self.ntraining, self.nonimplausible_bounds)

            elif self.shape == 'ellipsoid':
                # find mean and covariance of samples
                K0 = np.cov(nonimplausible_samples[:,:-1].T)
                mean = np.mean(nonimplausible_samples[:,:-1], axis=0)
                theta_train, theta_samples = sample.ellipsoid_sample(self.ndim, self.nsamples, self.ntraining, mean, K0)

            elif self.shape == 'hypercube_rot':
                    theta_train, theta_samples = sample.rotated_hypercube_samples(self.ndim, nonimplausible_samples[:,:-1], self.nsamples, self.ntraining)

            # store results
            nonimp_bounds.append(self.nonimplausible_bounds)
            nonimp_samples.append(nonimplausible_samples)
            I_samples_list.append(I_samples)
            emulator_output = np.append(emulator_output, np.concatenate((mu.reshape(-1,1), sd.reshape(-1,1)),axis=1), axis=0)

            cube_volume = utils.hypercube_volume(self.ndim, self.nonimplausible_bounds)
            print('Relative nonimplausible volume remaining: ' + str(round(cube_volume/initial_volume,3)))

        return Results({'nonimp_bounds': nonimp_bounds, 'nonimplausible': nonimp_samples, 'samples': sample_pts, 'training_pts': training_pts, 'emulator_output': emulator_output, 'I_samples': I_samples_list})


class Simulator(HistoryMatch): 

    """
    Simulator that generates outputs given an input of parameter values.
    Must be defined before running the history match.

    Args
    ----
    samples : ndarray, shape (N, ndim)
        Array of N input parameters.

    Returns
    -------
    output: ndarray, shape (N,)
        Array of the single simulator output given for each set of inputs.

    """


    def __init__(self, hm_instance):
        self.hm_instance = hm_instance

    def set_simulator(self, model):
        self.hm_instance.simulator = model

            

            
                