"""

Collection of helper functions.

"""

import numpy as np


def locate_boundaries(data, ndim):

    '''
    Args
    ----
    data : ndarray, shape (N, ndim) 
        Array of N datapoints.

    ndim : int
        Number of dimensions.
        
    Returns
    -------
    bounds : ndarray, shape (ndim, 2)
        Array of minimum and maximum values in each dimension.
        
    '''
    
    def find_minmax(data, i):
        return np.array([data[:,i].min(), data[:,i].max()]).reshape(1,-1)

    bounds = np.concatenate([find_minmax(data, i) for i in range(ndim)]) 

    return bounds
