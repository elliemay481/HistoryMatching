import numpy as np
from pyDOE import lhs
from scipy.stats import norm

def LHsampling(ndim, Ntraining, limits):
    
    '''
    Args:
    ndim : Number of dimensions
    Ntraining : Number of training points
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns: (d x M) array of training points
    
    '''
    
    # generate sample points
    input_train = lhs(ndim, samples=Ntraining, criterion='center')
    
    # adjust sample points for parameter ranges
    for i in range(ndim):
        input_train[:,i] = input_train[:,i]*(limits[i,1]-limits[i,0]) + limits[i,0]

    
    return input_train




def hypercube_sample(ndim, Nsamples, Ntraining, limits):

    '''
    Args:
    ndim : Number of dimensions
    Ntraining : Number of training points
    Nsamples : Number of points along parameter space axes
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns:
        (Ntraining x ndim) array of training points
        (Nsamples x ndim) array of well spaced samples in parameter space
    
    '''
    # generate sample points on Latin Hypercube
    input_test = LHsampling(ndim, Nsamples, limits)
    input_train = LHsampling(ndim, Ntraining, limits)

    return input_train, input_test


def ellipsoid_sample(ndim, data, Nsamples, Ntraining):

    '''
    Args:
    ndim : Number of dimensions
    nonimplausible_region : Nonimplausible sample points remaining at end
        of previous wave. Each column represents a parameter.
    Nsamples : Number of points along parameter space axes
    Ntraining : Number of training points
    
    Returns:
        (Ntraining x ndim) array of training points
        (Nsamples x ndim) array of well spaced samples in parameter space
    
    '''
    # generate sample points
    u_train = lhs(ndim, samples=Ntraining, criterion='center')
    u_test = lhs(ndim, samples=Nsamples, criterion='center')

    # normally distribute sample points
    for i in range(ndim):
        u_train[:,i] = norm(loc=0, scale=1).ppf(u_train[:,i])
        u_test[:,i] = norm(loc=0, scale=1).ppf(u_test[:,i])


    # find mean and covariance of samples
    K0 = np.cov(data.T)
    mean = np.mean(data, axis=0)

    # Add pertubation
    epsilon = 0.00001
    K = K0 + epsilon*np.identity(ndim)

    # Calculate the Cholesky decomposition
    L = np.linalg.cholesky(K)

    # Compute multivariate gaussian distributed samples
    input_train = mean.reshape(ndim, 1) + np.dot(L, u_train.reshape(ndim, Ntraining))
    input_test = mean.reshape(ndim, 1) + np.dot(L, u_test.reshape(ndim, Nsamples))

    return input_train.T, input_test.T