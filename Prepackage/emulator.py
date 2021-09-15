import numpy as np
from scipy.optimize import minimize

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
    def __init__(self, input_train, input_test, output_train, sigma, beta, kernel):
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train
        self.sigma = sigma
        self.kernel = kernel
        self.beta = beta
        



    def emulate(self):
        """
            
        """
        
        K_XX = self.kernel(self.input_train, self.input_train, self.sigma, self.theta)
        K_XsX = self.kernel(self.input_test, self.input_train, self.sigma, self.theta)
        K_XXs = self.kernel(self.input_train, self.input_test, self.sigma, self.theta)
        K_XsXs = self.kernel(self.input_test, self.input_test, self.sigma, self.theta)
        K_XX_inv = np.linalg.inv(K_XX)

        self.K_XX = K_XX
        self.K_XsX = K_XsX
        self.K_XXs = K_XXs
        self.K_XsXs = K_XsXs
        self.K_XX_inv = K_XX_inv

        mu = self.beta + self.K_XsX.dot(self.K_XX_inv).dot(self.output_train - self.beta)
        cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
        
        sd = np.sqrt(np.abs(np.diag(cov)))
        return mu, cov, sd



    def optimise(self):
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
                

        result = minimize(neg_log_marginal_likelihood, x0 = 3)
        self.theta = result.x
        print(self.theta)

    def __repr__(self): 
        return "Test a:% s b:% s" % (self.theta, self.sigma) 