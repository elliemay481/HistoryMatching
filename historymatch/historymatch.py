# external imports
import numpy as np
from tqdm import tqdm
import pickle

# internal imports
from historymatch import emulators
from historymatch import utils
from historymatch import sample

class Results(object):
    """
    For accessing the outputs of the history match.

    Args
    ----
    
    filename : str
        Name of pickle file containing results
    """

    def __init__(self, filename):

        self.filename = filename

        results_dict = {'wave': [], 'nonimplausible': [], 'samples_I': [], 'emulator_mu': [], 'emulator_sd': [], 'parameter_train1': []}
        self.results_dict = results_dict
        self.wave = self.results_dict['wave']
        self.nonimplausible = self.results_dict['nonimplausible']
        self.samples_I = self.results_dict['samples_I']
        self.emulator_mu = self.results_dict['emulator_mu']
        self.emulator_sd = self.results_dict['emulator_sd']
        self.parameter_train1 = self.results_dict['parameter_train1']

    def __getattribute__(self, name):
        """ Returns attribute of history matching results."""
        attr = object.__getattribute__(self, name)
        return attr

    def save_to_file(self):
        with open(self.filename, 'wb') as resultfile:
            pickle.dump(self.results_dict, resultfile)

    def get_nonimplausible(self, wave):
        with open(self.filename, 'rb') as resultfile:
            results_dict = pickle.load(resultfile)
        return results_dict['nonimplausible'][wave-1]



class HistoryMatch:

    """

    Args
    ----
    
    ndim : int
        Number of unknown parameters.

    emulator : {'GP', 'EC'}
        Type of emulator used to emulate the model. Choices are Gaussian process
        ('GP') and eigenvector continuation ('EC').

    volume_shape : {'hypercube', 'gaussian', 'hypercube_rot', 'ellipsoid'}
        Determines the shape of the nonimplausible parameter volume.
    
    nwaves : int, optional
        Number of history matching waves to complete. If None (default is None)
        scheme will continue for 20 waves or until termination conditions are met,
        whichever is sooner. Total number of waves may be less than nwaves if
        termination conditions reached earlier.

    ntraining : int, optional

    nsamples : int, optional

    """
    
    def __init__(self, ndim, filename='result_dict', emulator='GP', volume_shape='hypercube'):

        self.ndim = ndim

        self.simulator = None
        
        self.emulator_choice = emulator

        self.shape = volume_shape
        volume_types = ['gaussian', 'hypercube', 'hypercube_rot', 'ellipsoid']
        assert volume_shape in volume_types, \
                        "Invalid volume shape. Must be one of: 'gaussian', 'hypercube', 'hypercube_rot', 'ellipsoid'"
        
    def initialize_volume(self, min_parameter, max_parameter, ninactive=None, inactive_wave = None, sigma_inactive = None):

        '''
        Args
        ---
        min : ndarray, shape (ndim,)
        Array of the minimum parameter values to consider.

        max : ndarray, shape (ndim,)
        Array of the maximum parameter values to consider.
        
        '''

        if ninactive:
            self.ndim -= ninactive
            self.inactive = True
            self.ninactive = ninactive
            self.inactive_wave = inactive_wave
            self.sigma_inactive = sigma_inactive
        else:
            self.inactive = False
        self.parameter_bounds = np.concatenate((np.array(min_parameter).reshape(-1,1),\
                                                np.array(max_parameter).reshape(-1,1)), axis=1)
        

    def set_observations(self, obs_data, variables=None, sigma_obs=0, sigma_model=0,
                            sigma_method=0, sigma_inactive = None):

        '''
        Args
        ---

        obs_data : list of ndarray
        **** edit this ****
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
        self.variables = variables

        self.sigma_model = sigma_model
        #self.sigmas_mm = np.array([sigma_method, sigma_model])
        #assert not (all(s == 0 for s in self.sigmas_mm) and np.all(self.sigma_obs == 0)), \
                        #"At least one standard deviation must be nonzero."k
        self.obs_data = obs_data
        self.sigma_method = sigma_method
        self.all_sigma_model = sigma_model

        


    def simulate(self, parameters, variables_i=None):
        if self.simulator is None:
            raise NotImplementedError("Simulator not defined.")
        elif variables_i is None:
            return self.simulator(parameters)
        else:
            return self.simulator(parameters, *variables_i)

    def implausibility(self, E, z_i, var_em, var_obs, var_method, var_model, var_inactive):

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

        var_inactive : float
            Additional variance of uncertainty given the exclusion of one or more parameters.

        """
        

    
        return np.sqrt( ( E - z_i )**2  /  (var_em + var_obs + var_method + var_model + var_inactive) )


    def generate_samples(self, nonimplausible_samples, nsamples, ntraining, wave):

        '''
        Generate well-spaced samples using Latin Hypercube sampling and transform according
        to chosen volume shape.

        Args
        ---

        nonimplausible_samples : ndarray, shape (N, ndim)
            Array of N (non-implausible) samples from previous wave.

        Returns
        -------

        parameter_train : ndarray, shape (Ntraining, ndim)
            Input parameters used within simulator to generate outputs for emulator training.

        parameter_samples : ndarray, shape (Nsamples, ndim)
            Well spaced samples over non-implausible parameter space.
        
        '''
        
        if ntraining == None:
            ntraining = 2     # no emulator required so generate minimum amount

        # generate new well spaced samples in parameter space
        if self.shape == 'hypercube':
        # if converting inactive input to active
            if self.inactive == True and wave == self.inactive_wave:
                self.ndim += self.ninactive
                parameter_train, parameter_samples = \
                            sample.hypercube_sample(self.ndim, nsamples, ntraining, nonimplausible_samples,\
                                                    inactive=True, parameter_bounds=self.parameter_bounds)
            else:
                parameter_train, parameter_samples = \
                            sample.hypercube_sample(self.ndim, nsamples, ntraining, nonimplausible_samples)

        elif self.shape == 'gaussian':
            print('yes')
            if self.inactive == True and wave == self.inactive_wave:
                self.ndim += self.ninactive
                parameter_train, parameter_samples = \
                            sample.gaussian_sample(self.ndim, nsamples, ntraining, nonimplausible_samples, \
                                                    inactive=True, parameter_bounds=self.parameter_bounds)
            else:
                parameter_train, parameter_samples = \
                            sample.gaussian_sample(self.ndim, nsamples, ntraining, nonimplausible_samples)

        elif self.shape == 'ellipsoid':
            if self.inactive == True and wave == self.inactive_wave:
                self.ndim += self.ninactive
                parameter_train, parameter_samples = \
                            sample.uniform_ellipsoid_sample(self.ndim, nsamples, ntraining, nonimplausible_samples, \
                                                            inactive=True, parameter_bounds=self.parameter_bounds)
            else:
                parameter_train, parameter_samples = \
                            sample.uniform_ellipsoid_sample(self.ndim, nsamples, ntraining, nonimplausible_samples)

        elif self.shape == 'hypercube_rot':
            if self.inactive == True and wave == self.inactive_wave:
                self.ndim += self.ninactive
                parameter_train, parameter_samples = \
                            sample.rotated_hypercube_sample(self.ndim, nsamples, ntraining, nonimplausible_samples, \
                                                            self.parameter_bounds, inactive=True)
            else:
                parameter_train, parameter_samples = \
                            sample.rotated_hypercube_sample(self.ndim, nsamples, ntraining, nonimplausible_samples, \
                                                            self.parameter_bounds)

        return parameter_train, parameter_samples

    def run_wave(self, wave, observational_data, sigma_observational, sigma_model, sigma_method, wave_variables, nsamples, ntraining=None, emulate=False, Imax=2):


        """
        Performs a single wave of history matching.

        Args
        ----
        parameter_train : ndarray, shape (ntraining, ndim)
            Input parameters used within the simulator.

        parameter_samples : ndarray, shape (nsamples, ndim)
            Input parameters used within the emulator.

        Returns
        -------
        self.nonimplausible_volume, nonimplausible_samples, I_samples, mu, sd, output_convergence

        parameter_train : ndarray, shape (ntraining, ndim)
            Input parameters used within the simulator.

        """

        # store results
        training_pts = []

        # generate samples over parameter space
        if wave == 1:
            # initialise training set and parameter space
            parameter_train, parameter_samples = sample.hypercube_sample(self.ndim, nsamples, ntraining, parameter_bounds=self.parameter_bounds)
        else:
            # access nonimplausible samples from previous wave
            nonimplausible_samples = self.results.get_nonimplausible(wave-1)
            parameter_train, parameter_samples = self.generate_samples(nonimplausible_samples[:,:-1], nsamples, ntraining, wave)


        training_pts.append(parameter_train)
        
        noutputs = len(observational_data)
        # initialise arrays for storage
        self.output_convergence = np.full(noutputs, False, dtype=bool)
        implausibilities_all = np.zeros((nsamples, noutputs))

        for output in tqdm(range(noutputs)):

            # evaluate simulator over training points
            Ztrain = self.simulate(parameter_train.T, variables_i=wave_variables[output])

            if emulate == False:
                mu = self.simulate(parameter_samples.T, variables_i=wave_variables[output])
                sd = np.zeros(len(mu))
            else:
                assert not ntraining == None, \
                    "Must specify number of training points if using emulator."
                # train emulator
                if self.emulator_choice == 'GP':
                    GP = emulators.GaussianProcess(parameter_train, Ztrain, length_scale=10, signal_sd=100, bayes_linear = True, noise_sd = 1e-9)
                elif self.emulator_choice == 'EC':
                    print('EC not yet developed')
                # emulate outputs over sample space
                mu, sd = GP.emulate(parameter_samples)

            # check for emulator variance falling below other variances
            if np.mean(sd) + 3*np.sqrt(np.var(sd)) < sigma_model[output]:
                self.output_convergence[output] = True
            else:
                self.output_convergence[output] = False

            # once inactive parameter introduced, inactive uncertainty not neeeded
            if self.inactive == True and wave < self.inactive_wave:
                sigma_inactive = self.sigma_inactive
            else:
                sigma_inactive = 0

            # evaluate implausibility measure over parameter space
            for i in range(len(parameter_samples)):
                implausibilities_all[i, output] = self.implausibility(mu[i], observational_data[output], sd[i]**2, sigma_observational[output]**2,\
                                                                            sigma_method[output]**2, sigma_model[output]**2, sigma_inactive**2)

            if output == 0:
                GP = emulators.GaussianProcess(parameter_train, Ztrain, length_scale=10, signal_sd=100, bayes_linear = True, noise_sd = 1e-9)
                if wave == 1:
                    uniform_samples = np.zeros((100,4))
                    for i in range(4):
                        uniform_samples[:,i] = np.linspace(self.parameter_bounds[i,0], self.parameter_bounds[i,1], 100)
                    uniform_mu, uniform_sd = GP.emulate(uniform_samples)
                    emulator_mu = uniform_mu
                    emulator_sd = uniform_sd
                    parameter_train1 = parameter_train
                elif wave == 2:
                    bounds = utils.locate_boundaries(nonimplausible_samples[:,:-1], 4)
                    uniform_samples = np.zeros((100,4))
                    for i in range(4):
                        uniform_samples[:,i] = np.linspace(self.parameter_bounds[i,0], self.parameter_bounds[i,1], 100)
                    uniform_mu, uniform_sd = GP.emulate(uniform_samples)
                    emulator_mu = uniform_mu
                    emulator_sd = uniform_sd
                    parameter_train1 = parameter_train


        # get index of second highest maximum implaus for all outputs
        max_implausibilities = implausibilities_all[range(len(parameter_samples)), implausibilities_all.argsort()[:,-Imax]]
        # discard samples with implausibility over threshold
        samples_I = np.concatenate((parameter_samples, max_implausibilities.reshape(-1,1)), axis=1)
        nonimplausible_samples = np.delete(samples_I, np.where(samples_I[:,-1] > 3), axis=0)

        print('Number of Non-Implausible Samples: ' + str(nonimplausible_samples.shape[0]))

        return nonimplausible_samples, samples_I, emulator_mu, emulator_sd, parameter_train1



    def run(self, nwaves, ntraining, nsamples, result_obj=None, emulate=False):

        """
        Performs multiple waves of history matching to find nonimplausible parameter space.
        Requires observational data to be initialised beforehand.

        Returns
        -------
        result: 'Result'
            Dictionary containing results of each wave

        """

        if self.obs_data is None:
            raise NotImplementedError("Observational data not initialised.")
        if result_obj is None:
            Result = Results('historymatch_results')

        self.nwaves = nwaves

        for wave in np.arange(1,self.nwaves+1,1):
            print('Running wave ' + str(wave))

            # select data for wave
            observational_data = self.obs_data[wave-1]
            sigma_observational = self.sigma_obs[wave-1]
            sigma_model = self.sigma_model[wave-1]
            sigma_method = self.sigma_method[wave-1]
            wave_variables = self.variables[wave-1]


            nonimplausible_samples, samples_I, emulator_mu, emulator_sd, parameter_train1 = \
                            self.run_wave(wave, observational_data, sigma_observational,\
                                             sigma_model, sigma_method, wave_variables, nsamples, ntraining=ntraining, emulate=emulate)
            self.store_result(Result, wave, (nonimplausible_samples, samples_I, emulator_mu, emulator_sd, parameter_train1))
            
            # check for empty non-implausible volume
            if nonimplausible_samples.shape[0] == 0:
                end_str_empty = 'Nonimplausible volume empty. Terminated after ' + str(wave) + ' waves.'
                print(end_str_empty)
                return None
            # check for emulator variance lower than other variances
            elif np.all(self.output_convergence) == True and emulate == True:
                end_str_conv = 'Convergence reached after ' + str(wave) + ' waves.'
                print(end_str_conv)
                #return None

        return self.results

    def store_result(self, result_obj, wave, wave_results):
        
        # check for duplicate and overwrite
        if wave in result_obj.results_dict['wave']:
            all_nonimplausible = result_obj.results_dict['nonimplausible']
            all_samples = result_obj.results_dict['samples_I']
            all_nonimplausible[wave-1] = wave_results[0]
            all_samples[wave-1] = wave_results[1]
            result_obj.results_dict['nonimplausible'] = all_nonimplausible
            result_obj.results_dict['samples_I'] = all_samples

        else:   # store as most recent wave
            result_obj.results_dict['wave'].append(wave)
            result_obj.results_dict['nonimplausible'].append(wave_results[0])
            result_obj.results_dict['samples_I'].append(wave_results[1])

        result_obj.results_dict['emulator_mu'].append(wave_results[2])
        result_obj.results_dict['emulator_sd'].append(wave_results[3])
        result_obj.results_dict['parameter_train1'].append(wave_results[4])

        result_obj.save_to_file()
        # store for access in later waves
        self.results = result_obj
        


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

            

            
                