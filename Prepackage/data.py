import numpy as np
from pyDOE import lhs

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




def prepare_data(ndim, Nsamples, Ntraining, limits):

    '''
    Args:
    ndim : Number of dimensions
    Ntraining : Number of training points
    Nsamples : Number of points along parameter space axes
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns: (d x M) array of training points
    
    '''
    # generate sample points on Latin Hypercube
    input_test = LHsampling(ndim, Nsamples, limits)
    input_train = LHsampling(ndim, Ntraining, limits)

    return input_train, input_test





def locate_boundaries(data, ndim):

    '''
    Args:
    data: (n x ndim) array of data
    ndim : Number of dimensions
    Ntraining : Number of training points
    Nsamples : Number of points along parameter space axes
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns: (d x M) array of training points
    
    '''
    
    def find_minmax(data, i):
        return np.array([data[:,i].min(), data[:,i].max()]).reshape(1,-1)

    return np.concatenate([find_minmax(data, i) for i in range(ndim)]) 