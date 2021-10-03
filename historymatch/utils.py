"""

Collection of helper functions.

"""

import numpy as np

def hypercube_volume(ndim, bounds):
        volume = 1
        for i in range(ndim):
            volume *= bounds[i,1] - bounds[i,0]
        return volume



def locate_boundaries(data, ndim):

        '''
        Args:
        data: (n x ndim) array of data
        ndim : Number of dimensions
        
        Returns: (ndim x 2) array of minimum and maximum values for each parameter.
        
        '''
    
        def find_minmax(data, i):
            return np.array([data[:,i].min(), data[:,i].max()]).reshape(1,-1)

        return np.concatenate([find_minmax(data, i) for i in range(ndim)]) 