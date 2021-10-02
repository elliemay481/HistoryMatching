import numpy as np

def SE():
    """Create a squared exponential kernel

    Returns:
        squared_exponential (function) : A function of two numpy.ndarrays of floats that computes the squared exponential
        kernel with given standard deviation and length scale.
    """

    def squared_exponential(x1,x2,sigma,theta):
        """
        Args:
            x1 : (n x d) array of floats
            x2 : (m x d) array of floats
            sigma (float) : The standard deviation of the kernel.
            theta (float) : The length scale of the kernel.

        Returns:
            (n x m) covariance matrix
        """

        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)
        if x2.ndim == 1:
            x2 = x2.reshape(-1, 1)
        
        norm_sq = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1)  - 2 * np.dot(x1, x2.T)
        K = sigma**2 * np.exp(- norm_sq / ((theta**2)))
        return K
    
    return squared_exponential
