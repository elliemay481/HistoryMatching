# external imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import time

# internal imports
from historymatch import emulators
from historymatch import plot
from historymatch import utils
from historymatch import sample

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
            self.ntraining = 5

        if nsamples:
            self.nsamples = nsamples
        else:
            self.nsamples = 2 * (10**4)

        self.shape = volume_shape
        self.Z = None

    
    def save_model(self, output_filename):
        None

    def initialize_volume(self, min_theta, max_theta):

        '''
        Args
        ---
        min : ndarray, shape (ndim,)
        Array of the minimum parameter values to consider.

        max : ndarray, shape (ndim,)
        Array of the maximum parameter values to consider.
        
        '''

        self.nonimplausible_bounds = np.concatenate((np.array(min_theta).reshape(-1,1), np.array(max_theta).reshape(-1,1)), axis=1)

    def set_observations(self, obs_data, variables=None, sigma_obs=0, sigma_method=0,
                            sigma_model=0, outputs_per_wave = None ):

        '''
        Args
        ---

        obs_data : ndarray, shape (noutputs,)
        1-d array of observational data. Each element of the array corresponds to a model output.

        variables : ndarray, shape (nvariables, noutputs), optional
        Array of independent variables in simulator. Each row corresponds to an observational datapoint.

        sigma_method : float
            Standard deviation of method uncertainty

        sigma_model : float
            Standard deviation of model uncertainty

        sigma_obs : ndarray, shape (len(obs_data),)
            Standard deviation of the observational uncertainty for each observational datapoint provided.

        sigma_other : float
            Standard deviation of other uncertainties
        
        '''
        self.sigma_obs = sigma_obs
        self.sigma_model = sigma_model
        self.sigma_method = sigma_method
        #self.sigmas_mm = np.array([sigma_method, sigma_model])
        #assert not (all(s == 0 for s in self.sigmas_mm) and np.all(self.sigma_obs == 0)), \
                        #"At least one standard deviation must be nonzero."
        self.Z = obs_data
        self.sigma_method = sigma_method
        self.all_sigma_model = sigma_model
        if outputs_per_wave == None:
            self.noutputs_list = []
            self.noutputs = len(obs_data)
        else:
            self.noutputs_list = outputs_per_wave
            self.noutputs = self.noutputs_list[0]
        self.variables = variables


    def simulate(self, *theta, variables_i=None):
        if self.simulator is None:
            raise NotImplementedError("Simulator not defined.")
        elif self.variables is None:
            return self.simulator(*theta)
        else:
            return self.simulator(*theta, *variables_i)

    def implausibility(self, E, z_i, var_em, var_obs, var_method, var_model):

        """
        Evaluates the implausibility measure given emulator output and observational
        data.

        Args
        ----
        E : array
            Expectation values given as output from the emulator

        z_i : float
            Observational datapoint

        var_em: float
            Variance of emulator uncertainty

        var_method : float
            Variance of method uncertainty

        var_model : float
            Variance of model uncertainty

        var_obs : ndarray, shape (len(obs_data),)
            Variance of the observational uncertainty for output.

        """
        

    
        return np.sqrt( ( E - z_i )**2  /  (var_em + var_obs + var_method + var_model) )


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

        output_convergence = np.full(self.noutputs, False, dtype=bool)
        implausibilities_all = np.zeros((self.nsamples, self.noutputs))

        for output in tqdm(range(self.noutputs)):


            # evaluate simulator over training points
            Ztrain = self.simulate(*theta_train.T, variables_i=self.variables[output])
            
            #corr_length = self.nonimplausible_bounds[:,1]- self.nonimplausible_bounds[:,0]
            corr_length = 5
            
            if self.emulator_choice == 'GP':
                GP = emulators.GaussianProcess(theta_train, Ztrain, length_scale=corr_length, signal_sd=100, bayes_linear = True, noise_sd = 1e-9)
            elif self.emulator_choice == 'EC':
                print('EC not yet developed')

            #print('Emulating output {}...'.format(output))

            #mu = self.simulate(*theta_samples.T, variables_i=self.variables[output])

            #sd = np.zeros(len(mu))
            
            mu, sd = GP.emulate(theta_samples)


            if np.mean(sd) + 3*np.sqrt(np.var(sd)) < self.sigma_model[output]:
                output_convergence[output] = True
            else:
                output_convergence[output] = False

            for i in range(len(theta_samples)):
                implausibilities_all[i, output] = self.implausibility(mu[i], self.Z[output], sd[i]**2, self.sigma_obs[output]**2,\
                                                                        self.sigma_method**2, self.sigma_model[output]**2)


        # get index of second highest maximum implaus for all outputs
        if self.noutputs < 2:
            max_I = implausibilities_all.argsort()[:,-1]
            max_implausibilities = implausibilities_all[range(len(max_I)), max_I]
        else:
            max2_I = implausibilities_all.argsort()[:,-2]
            print(max2_I)
            max_implausibilities = implausibilities_all[range(len(max2_I)), max2_I]

        I_samples = np.concatenate((theta_samples, max_implausibilities.reshape(-1,1)), axis=1)
        nonimplausible_samples = np.delete(I_samples, np.where(I_samples[:,-1] > 3), axis=0)

        print('Number of Non-Implausible Samples : ' + str(nonimplausible_samples.shape))
        if self.shape == 'hypercube':
            self.nonimplausible_bounds = utils.locate_boundaries(nonimplausible_samples, self.ndim)
        elif self.shape == 'hypercube_rot':
            # ***** fix **********
            #training_rot, samples_rot = sample.rotated_hypercube_samples(self.ndim, nonimplausible_samples[:,:-1], self.nsamples, self.ntraining)
            self.nonimplausible_bounds = utils.locate_boundaries(nonimplausible_samples[:,:-1], self.ndim)
        else:
            # ****** fix **********
            self.nonimplausible_bounds = utils.locate_boundaries(nonimplausible_samples, self.ndim)
        print(self.nonimplausible_bounds)
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

        print('Number of Samples : ' + str(self.nsamples))

        # initialise training set and parameter space
        theta_train, theta_samples = sample.hypercube_sample(self.ndim, self.nsamples, self.ntraining, self.nonimplausible_bounds)
        #theta_train = sample.grid_sample(self.ndim, self.ntraining, self.nonimplausible_bounds)
        #theta_samples = sample.grid_sample(self.ndim, self.nsamples, self.nonimplausible_bounds)
        
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


            if wave+1 < len(self.noutputs_list):
                self.noutputs = self.noutputs_list[wave+1]
            else:
                self.noutputs = len(self.Z)

            print('Convergence : ' + str(np.all(output_convergence)))
            #if np.all(output_convergence) = True:
                #end_str_conv = 'Convergence reached after ' + str(wave) + ' waves.'
                #return end_str_conv
            
            if nonimplausible_samples.shape[0] == 0:
                end_str_empty = 'Nonimplausible volume empty. Terminated after ' + str(wave) + ' waves.'
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
    parameters : ndarray, shape (N, ndim)
        Array of N sets of input parameters.

    variables : ndarray, shape (m,)
        Array of other independent variables in model

    Returns
    -------
    output: ndarray, shape (N,)
        Array of the single simulator output given for each set of inputs.

    """


    def __init__(self, hm_instance):
        self.hm_instance = hm_instance

    def set_simulator(self, model):
        self.hm_instance.simulator = model

            

            
                