import numpy as np
from . import *

class Gaussian_Process:
    """
    Args:
        x_train (numpy.ndarray n x d) : Input training data
        x_test (numpy.ndarray m x d) : Input validation data
        y_train (numpy.ndarray n x d) : Output training data
        sigma (float) : The standard deviation of the kernel.
        theta (float) : The length scale of the kernel.
        beta (float) : Prior expectation
        kernel (function)

    """
    def __init__(self, x_train, x_test, y_train, beta, kernel):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.beta = beta
        self.kernel = kernel

        K_XX = kernel(x_train, x_train)
        K_XsX = kernel(x_test, x_train)
        K_XXs = kernel(x_train, x_test)
        K_XsXs = kernel(x_test, x_test)
        K_XX_inv = np.linalg.inv(K_XX)

        self.K_XX = K_XX
        self.K_XsX = K_XsX
        self.K_XXs = K_XXs
        self.K_XsXs = K_XsXs
        self.K_XX_inv = K_XX_inv


    def emulate(self):
        """
            
        """
        
        mu = self.beta + self.K_XsX.dot(self.K_XX_inv).dot(self.y_train - self.beta)
        cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
        
        sd = np.sqrt(np.diag(cov))
    
        return mu, cov, sd