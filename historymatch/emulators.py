
from abc import ABC, abstractmethod
import numpy as np
from math import factorial
#from scipy.optimize import minimize
#from scipy.linalg import solve_triangular
#from scipy.optimize import basinhopping


class Emulator(ABC):

    @abstractmethod
    def emulate(self, param_samples):
        pass

class GaussianProcess(Emulator):

    def __init__(self, input_train, output_train, length_scale, signal_sd=0.1, noise_sd = None, ols_order = 1, bayes_linear = True):
        self.sigma_f = signal_sd
        self.sigma_noise = noise_sd
        self.l = length_scale
        self.input_train = input_train
        self.output_train = output_train
        self.ndim = self.input_train.shape[1]
        self.kernel = self.SE()
        self.order = ols_order
        self.bayes_linear = bayes_linear



    def emulate(self, param_samples):

        if self.bayes_linear == True:
            # perform linear regression
            coeff_ols, train_ols, var_ols = self.linear_regression()
            self.coeff_ols = coeff_ols
            self.var_ols = var_ols
            self.train_ols = train_ols
            self.sigma_f = np.sqrt(var_ols)

        self.input_test = param_samples


        if self.sigma_noise != None:
            K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l) + (self.sigma_noise**2)*np.eye(len(self.input_train))
        else:
            K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l)
        K_XX_inv = np.linalg.inv(K_XX)

        self.K_XX = K_XX
        self.K_XX_inv = K_XX_inv
        
        K_XsX = self.kernel(self.input_test, self.input_train, self.sigma_f, self.l)
        K_XXs = self.kernel(self.input_train, self.input_test, self.sigma_f, self.l)
        K_XsXs = self.kernel(self.input_test, self.input_test, self.sigma_f, self.l)

        self.K_XsX = K_XsX
        self.K_XXs = K_XXs
        self.K_XsXs = K_XsXs

        

        if self.bayes_linear == True:
            Xd = self.design_matrix(self.input_test)
            mu_ols = np.dot(Xd, self.coeff_ols)

            # emulate
            mu = mu_ols + self.K_XsX.dot(self.K_XX_inv).dot(self.output_train - np.mean(self.output_train))
            cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)

            #print(self.sigma_f)
            #print(np.sqrt(np.var(self.output_train)))
            variance = np.abs(np.diag(cov))
            sd = np.sqrt(variance)
            print(sd)
            #self.var_ols + self.sigma_f**2 

        else:
            mu = self.K_XsX.dot(self.K_XX_inv).dot(self.output_train)
            cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
            sd = np.sqrt(np.abs(np.diag(cov)))


            #print(np.sqrt(np.abs(self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs))))
        return mu, sd


    def SE(self):
        """Create a squared exponential kernel

        Returns:
            squared_exponential (function) : A function of two numpy.ndarrays of floats that computes the squared exponential
            kernel with given standard deviation and length scale.
        """

        def squared_exponential(x1,x2,sigma_f,l):
            """
            Args:
                x1 : (n x d) array of floats
                x2 : (m x d) array of floats
                sigma (float) : The standard deviation of the kernel.
                l (float) : The length scale of the kernel.

            Returns:
                (n x m) covariance matrix
            """

            if x1.ndim == 1:
                x1 = x1.reshape(-1, 1)
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            
            norm_sq = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1)  - 2 * np.dot(x1, x2.T)
            K = sigma_f**2 * np.exp(- norm_sq / ((2*l**2)))
            return K
        
        return squared_exponential


    def linear_regression(self):

        """Performs linear regression

        Returns:
            coeff : 
                The regression coefficients
            mu_train : 
                The regression fit to the training samples.
            var : 
                The variance of the fit mu_train
        """

        X = self.design_matrix(self.input_train)
        #print(X)

        def solve(A, y):
            coeff = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(y)
            return coeff.flatten()

        coeff = solve(X,self.output_train)

        mu_train = np.dot(X, coeff)

        #print(self.output_train)
        #print(mu_train)
        

        var = np.dot((self.output_train - mu_train).T, (self.output_train - mu_train)) / len(mu_train)

        #print(np.sqrt(var))
        return coeff, mu_train, var


    '''
    def design_matrix(self, x):
            
            N = self.ndim*(self.order) + 1
            if self.order > 1:
                ncombinations = int(factorial(self.ndim) / (2*factorial(self.ndim-2)))
                X_d = np.zeros((len(x),N+ncombinations))
            else:
                X_d = np.zeros((len(x),N))
            X_d[:,0] = 1
            for i in range(self.order):
                for j in range(self.ndim):
                    X_d[:,1 + (i*self.ndim + j)] = x[:,j]**(i+1)
            prod = 0
            if self.order > 1:
                for i in range(self.ndim):
                    for j in range(self.ndim):
                        if i < j:
                            X_d[:,N + prod] = x[:,i]*x[:,j]
                            prod += 1
            return X_d'''

    def design_matrix(self, x):
        X_d = np.zeros((len(x),1))
        X_d[:,0] = 1
        return X_d

    

class EigenvectorContinuation(Emulator):

    def emulate(self):
        pass
