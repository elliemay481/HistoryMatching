# external imports
import numpy as np
from pyDOE import lhs
from scipy.stats import norm
from scipy.stats import chi2

# internal imports
from historymatch import utils



def LHsampling(ndim, Nsamples, limits):
    
    '''
    Generate well spaced samples using Latin Hypercube sampling.

    Args
    ---

    ndim : int
        Number of dimensions

    Nsamples : int
        Number of samples to generate.

    limits : ndarray, shape (ndim, 2)
        Array of upper and lower boundaries to generate samples between.

    Returns
    -------

    samples : ndarray, shape (Nsamples, ndim)
        Latin hypercube samples.
    
    '''
    
    # generate sample points
    samples = lhs(ndim, samples=Nsamples, criterion='center')
    # adjust sample points for parameter ranges
    if ndim == 1:
        
        samples = samples*(limits[0][1]-limits[0][0]) + limits[0][0]
    else:
        for i in range(ndim):
            samples[:,i] = samples[:,i]*(limits[i,1]-limits[i,0]) + limits[i,0]

    return samples




def hypercube_sample(ndim, Nsamples, Ntraining, nonimplausible_samples=None, inactive=False, parameter_bounds=None):

    '''
    Generate well-spaced samples distributed within a hyperrectangle.

    Args
    ---

    ndim : int
        Number of dimensions

    Ntraining : int
        Number of training points to generate.

    Nsamples : int
        Total number of samples to generate.

    bounds : ndarray, shape (2, ndim)
        Array of upper and lower boundaries of each input parameter.

    Returns
    -------

    parameter_train : ndarray, shape (Ntraining, ndim)
        Input parameters used within simulator to generate outputs for emulator training.

    parameter_samples : ndarray, shape (Nsamples, ndim)
        Well spaced samples in parameter space.
    
    '''

    # wave 1
    if type(nonimplausible_samples) is not np.ndarray:
        bounds = parameter_bounds
    else:
        if inactive == True:
            bounds = utils.locate_boundaries(nonimplausible_samples, ndim-1)
        else:
            bounds = utils.locate_boundaries(nonimplausible_samples, ndim)

    if inactive == True:
        extended_bounds = np.concatenate((bounds, parameter_bounds[-1].reshape(1,-1)),axis=0)
        bounds = extended_bounds


    # generate sample points on Latin Hypercube
    parameter_samples = LHsampling(ndim, Nsamples, bounds)
    parameter_train = LHsampling(ndim, Ntraining, bounds)

    return parameter_train, parameter_samples




def gaussian_sample(ndim, Nsamples, Ntraining, nonimplausible_samples, inactive=False, parameter_bounds=None):

    '''
    Generate well-spaced samples distributed according to a multivariate normal distribution.

    Args
    ---

    ndim : int
        Number of dimensions

    Ntraining : int
        Number of training points to generate.

    Nsamples : int
        Total number of samples to generate.

    mean : ndarray, shape (ndim,)
        Mean of normal distribution.

    covariance : ndarray, shape (ndim, ndim)
        Covariance matrix of distribution.

    Returns
    -------

    parameter_train : ndarray, shape (Ntraining, ndim)
        Input parameters used within simulator to generate outputs for emulator training.

    parameter_samples : ndarray, shape (Nsamples, ndim)
        Well spaced samples in parameter space.
    
    '''

    # compute mean and covariance of non-implausible samples
    covariance = np.cov(nonimplausible_samples.T)
    mean = np.mean(nonimplausible_samples, axis=0)

    # generate sample points
    uniform_train = lhs(ndim, samples=Ntraining, criterion='center')
    uniform_samples = lhs(ndim, samples=Nsamples, criterion='center')

    if inactive == True:
        ndim -= 1

    # normally distribute sample points
    for i in range(ndim):
        uniform_train[:,i] = norm(loc=0, scale=1).ppf(uniform_train[:,i])
        uniform_samples[:,i] = norm(loc=0, scale=1).ppf(uniform_samples[:,i])
    
    # Add pertubation to covariance
    epsilon = 1e-9
    K = covariance + epsilon*np.identity(ndim)

    # Compute the Cholesky decomposition
    L = np.linalg.cholesky(K)

    # Compute multivariate gaussian distributed samples
    input_train = mean.reshape(ndim, 1) + np.dot(L, uniform_train[:,:ndim].reshape(ndim, Ntraining))
    parameter_samples = mean.reshape(ndim, 1) + np.dot(L, uniform_samples[:,:ndim].reshape(ndim, Nsamples))

    if inactive == True:
        uniform_train[:,-1] = uniform_train[:,-1]*(parameter_bounds[-1,1]-parameter_bounds[-1,0]) + parameter_bounds[-1,0]
        uniform_samples[:,-1] = uniform_samples[:,-1]*(parameter_bounds[-1,1]-parameter_bounds[-1,0]) + parameter_bounds[-1,0]
        input_train_inac = np.concatenate((input_train.T, uniform_train[:,-1].reshape(-1,1)),axis=1)
        parameter_samples_inac = np.concatenate((parameter_samples.T, uniform_samples[:,-1].reshape(-1,1)),axis=1)
        return input_train_inac, parameter_samples_inac

    return input_train.T, parameter_samples.T




def uniform_ellipsoid_sample(ndim, Nsamples, Ntraining, nonimplausible_samples, inactive=False, parameter_bounds=None):

    '''
    Generate well-spaced samples distributed uniformly in an ellipsoid.

    Args
    ---

    ndim : int
        Number of dimensions

    Ntraining : int
        Number of training points to generate.

    Nsamples : int
        Total number of samples to generate.

    mean : ndarray, shape (ndim,)
        Mean of normal distribution.

    covariance : ndarray, shape (ndim, ndim)
        Covariance matrix of distribution.

    Returns
    -------

    parameter_train : ndarray, shape (Ntraining, ndim)
        Input parameters used within simulator to generate outputs for emulator training.

    parameter_samples : ndarray, shape (Nsamples, ndim)
        Well spaced samples in parameter space.
    
    '''

    # compute mean and covariance of non-implausible samples
    covariance = np.cov(nonimplausible_samples.T)

    mean = np.mean(nonimplausible_samples, axis=0)

    # generate sample points
    uniform_train = lhs(ndim, samples=Ntraining, criterion='center')
    uniform_samples = lhs(ndim, samples=Nsamples, criterion='center')

    if inactive == True:
        ndim -= 1

    # normally distribute sample points
    normal_train = np.zeros_like(uniform_train)
    normal_samples = np.zeros_like(uniform_samples)
    for i in range(ndim):
        normal_train[:,i] = norm(loc=0, scale=1).ppf(uniform_train[:,i])
        normal_samples[:,i] = norm(loc=0, scale=1).ppf(uniform_samples[:,i])
    
    # compute norm and divide by
    train_norm = np.linalg.norm(normal_train[:,:ndim], axis=1)
    sample_norm = np.linalg.norm(normal_samples[:,:ndim], axis=1)

    uniform_ell_train = np.zeros_like(normal_train[:,:ndim])
    uniform_ell_samples = np.zeros_like(normal_samples[:,:ndim])
    for i in range(ndim):
        uniform_ell_train[:,i] = normal_train[:,i]/train_norm
        uniform_ell_samples[:,i] = normal_samples[:,i]/sample_norm

    # radially distribute samples
    r_samp = np.zeros((Nsamples,ndim))
    r_train = np.zeros((Ntraining,ndim))
    for i in range(ndim):
            r_samp[:,i] = np.linspace(0,1,Nsamples)
            r_train[:,i] = np.linspace(0,1,Ntraining)

    uniform_ell_train *= r_train**(1.0/ndim)
    uniform_ell_samples *= r_samp**(1.0/ndim)

    # Add pertubation to covariance (for numerical stability)
    epsilon = 1e-9
    K = covariance + epsilon*np.identity(ndim)

    # Compute the Cholesky decomposition
    L = np.linalg.cholesky(K)

    # correlate samples
    ell_train = np.dot(L, uniform_ell_train.T)
    ell_samples = np.dot(L, uniform_ell_samples.T)

    # recentre and resize to 95% C.I.
    chisq = chi2.ppf(0.95, ndim) # 95% C.I. chisq values
    input_train = mean.reshape(ndim, 1) + np.sqrt(chisq)*ell_train
    parameter_samples = mean.reshape(ndim, 1) + np.sqrt(chisq)*ell_samples
    
    if inactive == True:
        uniform_train[:,-1] = uniform_train[:,-1]*(parameter_bounds[-1,1]-parameter_bounds[-1,0]) + parameter_bounds[-1,0]
        uniform_samples[:,-1] = uniform_samples[:,-1]*(parameter_bounds[-1,1]-parameter_bounds[-1,0]) + parameter_bounds[-1,0]
        input_train_inac = np.concatenate((input_train.T, uniform_train[:,-1].reshape(-1,1)),axis=1)
        parameter_samples_inac = np.concatenate((parameter_samples.T, uniform_samples[:,-1].reshape(-1,1)),axis=1)
        return input_train_inac, parameter_samples_inac

    return input_train.T, parameter_samples.T




def rotated_hypercube_sample(ndim, Nsamples, Ntraining, nonimplausible_samples, parameter_bounds=None, inactive=False):
    '''
    Generate well-spaced samples using Latin Hypercube sampling and rotate to correspond to non-implausible volume.

    Args
    ---

    ndim : int
        Number of dimensions

    Ntraining : int
        Number of training points to generate.

    Nsamples : int
        Total number of samples to generate.

    nonimplausible_samples : ndarray, shape (N, ndim)
        Array of N (non-implausible) samples.

    Returns
    -------

    parameter_train : ndarray, shape (Ntraining, ndim)
        Input parameters used within simulator to generate outputs for emulator training.

    parameter_samples : ndarray, shape (Nsamples, ndim)
        Well spaced samples in parameter space.
    
    '''
    # compute transformation matrices from nonimplausible samples
    covariance = np.cov(nonimplausible_samples.T)
    mean = np.mean(nonimplausible_samples, axis=0)

    # generate well spaced samples centred on 0
    bounds = np.concatenate((-np.ones(ndim).reshape(-1,1), np.ones(ndim).reshape(-1,1)), axis=1)
    uniform_train = LHsampling(ndim, Ntraining, bounds)
    uniform_samples = LHsampling(ndim, Nsamples, bounds)

    if inactive == True:
        ndim -= 1

    # rescale covariance to obtain correct scaling values
    chisq = chi2.ppf(0.95, ndim) # 95% C.I. chisq value
    covariance *= chisq

    _eigvals, eigvecs = np.linalg.eig(covariance)
    R = eigvecs.T                            # rotation matrix
    S = np.eye(ndim)*np.sqrt(_eigvals)     # scaling matrix
    T = np.dot(S, np.linalg.inv(R))          # tranformation matrix

    # rotate and scale samples
    parameter_train = np.zeros((len(uniform_train),ndim))
    parameter_samples = np.zeros((len(uniform_samples),ndim))
    for i in range(len(uniform_train)):
        parameter_train[i] = np.dot(np.linalg.inv(R), np.dot(S, uniform_train[i,:ndim]))
    for i in range(len(uniform_samples)):
        parameter_samples[i] = np.dot(np.linalg.inv(R), np.dot(S, uniform_samples[i,:ndim]))

    # recentre samples
    for dim in range(ndim):
        parameter_samples[:,dim] += mean[dim]

    # if inactive parameter introduced, append well spaced samples
    if inactive == True:
        uniform_train[:,-1] = 0.5*(uniform_train[:,-1]+1)*(parameter_bounds[-1,1]-parameter_bounds[-1,0]) + parameter_bounds[-1,0]
        uniform_samples[:,-1] = 0.5*(uniform_samples[:,-1]+1)*(parameter_bounds[-1,1]-parameter_bounds[-1,0]) + parameter_bounds[-1,0]
        parameter_train_inac = np.concatenate((parameter_train, uniform_train[:,-1].reshape(-1,1)),axis=1)
        parameter_samples_inac = np.concatenate((parameter_samples, uniform_samples[:,-1].reshape(-1,1)),axis=1)
        return parameter_train_inac, parameter_samples_inac

    return parameter_train, parameter_samples