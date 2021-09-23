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

def LHsampling_grid(ndim, Ntraining, limits, ncells):
    
    '''
    Args:
    ndim : Number of dimensions
    Nsamples : Total number of sample points
    limits: (n x d) array of upper and lower boundaries of parameter region
    ncells: Number of cells within cluster
    
    Returns: (d x M) array of training points
    
    '''

    n = int(np.ceil(Ntraining / ncells))
    input_train = np.empty((0, ndim))

    for cell in range(ncells):
        input_train_cell = lhs(ndim, samples=n)

        # adjust sample points for parameter ranges
        for i in range(ndim):
            input_train_cell[:,i] = input_train_cell[:,i]*(limits[cell][i,1]-limits[cell][i,0]) + limits[cell][i,0]

        input_train = np.append(input_train, input_train_cell, axis=0)

    return input_train

def prepare_data_old(ndim, Nsamples, Ntraining, limits):

    '''
    Args:
    ndim : Number of dimensions
    Ntraining : Number of training points
    Nsamples : Number of points along parameter space axes
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns: (d x M) array of training points
    
    '''

    parameter_space = np.zeros((Nsamples, ndim))
    # define parameter space
    args = []
    for i in range(ndim):
        parameter_space[:,i] = np.linspace(limits[i,0], limits[i,1], Nsamples)

    #input_test = np.array(np.meshgrid(*parameter_space.T)).T.reshape(-1,ndim)
    input_test = LHsampling(ndim, Nsamples, limits)
    # generate training points on Latin Hypercube
    input_train = LHsampling(ndim, Ntraining, limits)

     # evaluate true model over training inputs
    #output_train = np.zeros(Ntraining)
    #for i in range(Ntraining):
    #output_train = true_model_vec(*input_train.T)


    return input_train, input_test

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
    
    def find_minmax(data, i):
        return np.array([data[:,i].min(), data[:,i].max()]).reshape(1,-1)

    return np.concatenate([find_minmax(data, i) for i in range(ndim)]) 