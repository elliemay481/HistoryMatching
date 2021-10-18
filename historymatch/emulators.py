
from abc import ABC, abstractmethod
import numpy as np
from math import factorial
from scipy.optimize import minimize
from numpy.linalg import cholesky, det, inv
from scipy.linalg import solve_triangular
#from scipy.linalg import solve_triangular


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
        #self.w = 0.8

        if self.bayes_linear == True:
            # perform linear regression
            coeff_ols, train_ols, var_ols = self.linear_regression()
            self.coeff_ols = coeff_ols
            self.var_ols = var_ols
            self.train_ols = train_ols
            self.sigma_f = np.sqrt(var_ols)


    def emulate(self, param_samples):

        
        #self.sigma_noise = np.sqrt((1-self.w)*var_ols)

        self.input_test = param_samples


        #if self.sigma_noise != None:
        K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l) + (self.sigma_noise**2)*np.eye(len(self.input_train))
        #else:
        #K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l)
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


            variance = np.abs(np.diag(cov))
            sd = np.sqrt(variance)

        else:
            mu = self.K_XsX.dot(self.K_XX_inv).dot(self.output_train)
            cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
            sd = np.sqrt(np.abs(np.diag(cov)))

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

    
    def optimize(self):
        """
        Optimise GP hyperparameters

        Returns:
        

        """
        '''
        def neg_log_marginal_likelihood(sigma_n):
            
            #print(sigma_n)
            #print(self.var_ols)
            sigma_noise = sigma_n[0]
            #print(self.var_ols - sigma_noise**2)
            #print(sigma_noise)
            sigma_f = np.sqrt(self.var_ols - sigma_noise**2)

            K_XX = self.kernel(self.input_train, self.input_train, sigma_f, self.l) + (sigma_noise**2)*np.eye(len(self.input_train))

            K_XX_inv = np.linalg.inv(K_XX)

            #if np.linalg.det(K_XX) == 0:
                #return 1
            #else:
            return 0.5 * ( (self.output_train.T).dot((K_XX_inv.dot(self.output_train))) + np.log(np.linalg.det(K_XX)) + len(self.input_train)*np.log(2*np.pi) )'''
        
        def neg_log_marginal_likelihood(sigma_n):

            sigma_noise = sigma_n[0]
            #print(self.var_ols - sigma_noise**2)
            #print(sigma_noise)
            sigma_f = np.sqrt(self.var_ols - sigma_noise**2)

            K = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l) + (sigma_noise**2)*np.eye(len(self.input_train))
            y = self.output_train
            n = len(y)

            L = np.linalg.cholesky(K)
            alpha_0 = solve_triangular(L, y, lower=True)
            alpha = solve_triangular(L.T, alpha_0, lower=False)

            # check lower/upper triangles
            # check if y needs to be transposed


            #print(-0.5*(np.dot(y,alpha)) - np.sum(np.log(np.diagonal(L))) - 0.5*n*np.log(2*np.pi))
            return 0.5*(np.dot(y,alpha)) + np.sum(np.log(np.diagonal(L))) + 0.5*n*np.log(2*np.pi)
        
        bounds = [1e-9, np.sqrt(self.var_ols)-1e-9]
        n_list = []
        for i in range(10):
            n_init = (bounds[1] - bounds[0]) * np.random.random() + bounds[0]

            result = minimize(neg_log_marginal_likelihood, x0 = [n_init], bounds=[bounds], method='L-BFGS-B')
            n_list.append(result.x[0])

        self.sigma_noise = np.mean(n_list)

        self.sigma_f = np.sqrt(self.var_ols - self.sigma_noise**2)
        
        

class EigenvectorContinuation(Emulator):

    def emulate(self):
        pass
