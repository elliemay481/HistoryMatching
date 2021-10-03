import numpy as np
from scipy.optimize import minimize

class Gaussian_Process:
    """
    Args:
        x_train (numpy.ndarray n x d) : Input training data
        x_test (numpy.ndarray m x d) : Input of well-spaced samples
        y_train (numpy.ndarray n x d) : Output training data
        sigma (float) : The standard deviation of the kernel
        theta (float) : The length scale of the kernel.
        beta (float) : Prior expectation
        kernel (function)
        noise (bool)
        sigma_n (float) : standard devation of gaussian noise

    """
    def __init__(self, input_train, input_test, output_train, sigma=0.1, beta=0, theta=1, noise=False, sigma_n=None):
        self.sigma = sigma
        self.kernel = self.SE()
        self.beta = beta
        self.theta = theta
        self.noise = noise
        self.sigma_n = sigma_n
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train




    def emulate(self):
        """
            
        """

        
        if self.noise == True:
            K_XX = self.kernel(self.input_train, self.input_train, self.sigma, self.theta) + (self.sigma_n**2)*np.eye(len(self.input_train))
        else:
            K_XX = self.kernel(self.input_train, self.input_train, self.sigma, self.theta)
        K_XsX = self.kernel(self.input_test, self.input_train, self.sigma, self.theta)
        K_XXs = self.kernel(self.input_train, self.input_test, self.sigma, self.theta)
        K_XsXs = self.kernel(self.input_test, self.input_test, self.sigma, self.theta)
        K_XX_inv = np.linalg.inv(K_XX)

        #print(K_XX)
        #print(np.linalg.det(K_XX))

        self.K_XX = K_XX
        self.K_XsX = K_XsX
        self.K_XXs = K_XXs
        self.K_XsXs = K_XsXs
        self.K_XX_inv = K_XX_inv

        mu = self.beta + self.K_XsX.dot(self.K_XX_inv).dot(self.output_train - self.beta)
        cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
        
        sd = np.sqrt(np.abs(np.diag(cov)))
        return mu, cov, sd



    def optimize(self):
        """
        Optimise GP hyperparameters
        *** Only for length scale now ***


        Returns:
        

        """


        def neg_log_marginal_likelihood(theta):

            K_XX = self.kernel(self.input_train, self.input_train, self.sigma, theta)

            K_XX_inv = np.linalg.inv(K_XX)

            if np.linalg.det(K_XX) == 0:
                return 1
            else:
                return 0.5 * ( (self.output_train.T).dot((K_XX_inv.dot(self.output_train))) + np.log(np.linalg.det(K_XX)) + len(self.input_train)*np.log(2*np.pi) )
                

        result = minimize(neg_log_marginal_likelihood, x0 = 1)
        self.theta = result.x




    def SE(self):
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
