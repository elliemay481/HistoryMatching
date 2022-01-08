from abc import ABC, abstractmethod
import numpy as np
from math import factorial
import GPy


class Emulator(ABC):

    """

    Emulates the model output given a set of input parameters.

    Args
    ----
    samples : ndarray, shape (N, ndim)
        Array of N input parameters.

    Returns
    -------
    mu: ndarray, shape (N,)
        Array of the single emulator output given for each set of inputs.

    sd: ndarray, shape (N,)
        Standard deviation of each emulator output.

    """

    @abstractmethod
    def emulate(self, samples):
        pass

class GaussianProcess(Emulator):

    def __init__(self, input_train, output_train, length_scale, signal_sd=0.1, noise_sd = None, ols_order = 0, bayes_linear = True):
        self.sigma_f = signal_sd
        self.sigma_noise = noise_sd
        self.l = length_scale
        self.input_train = input_train
        self.output_train = output_train
        self.ndim = self.input_train.shape[1]
        self.kernel = self.SE()
        self.order = ols_order
        self.bayes_linear = bayes_linear
        
        if self.bayes_linear == True:
            # perform linear regression
            coeff_ols, train_ols, var_ols = self.linear_regression()
            self.coeff_ols = coeff_ols
            self.var_ols = var_ols
            self.train_ols = train_ols
            self.sigma_f = np.sqrt(var_ols)
            #print(self.sigma_f)

        
        '''
        K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l) + (self.sigma_noise**2)*np.eye(len(self.input_train))
        K_XX_inv = np.linalg.inv(K_XX)

        self.K_XX = K_XX
        self.K_XX_inv = K_XX_inv
        '''
        
        kern = GPy.kern.RBF(input_dim=self.ndim, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(input_train,output_train.reshape(-1,1),kern)
        m.optimize(messages=False)
        self.m = m




    def emulate(self, param_samples):

        
        self.input_test = param_samples

        #K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l) + (self.sigma_noise**2)*np.eye(len(self.input_train))
        #K_XX_inv = np.linalg.inv(K_XX)

        #self.K_XX = K_XX
        #self.K_XX_inv = K_XX_inv
        '''
        K_XsX = self.kernel(self.input_test, self.input_train, self.sigma_f, self.l)
        K_XXs = self.kernel(self.input_train, self.input_test, self.sigma_f, self.l)
        #K_XsXs = self.kernel(self.input_test, self.input_test, self.sigma_f, self.l)

        self.K_XsX = K_XsX
        self.K_XXs = K_XXs
        #self.K_XsXs = K_XsXs

        

        if self.bayes_linear == True:
            Xd = self.design_matrix(self.input_test)
            mu_ols = np.dot(Xd, self.coeff_ols)

            mu = mu_ols + self.K_XsX.dot(self.K_XX_inv).dot(self.output_train - np.mean(self.output_train))
            #cov = self.sigma_f**2 - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
            cov_diag = self.sigma_f**2 - np.einsum('ij,ji->i', np.dot(self.K_XsX, self.K_XX_inv), self.K_XXs)

            sd = np.sqrt(np.abs(cov_diag))
            #print(sd)

        else:
            mu = self.K_XsX.dot(self.K_XX_inv).dot(self.output_train)
            cov = self.K_XsXs - self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)
            sd = np.sqrt(np.abs(np.diag(cov)))'''

        

        mu, sd = self.m.predict(param_samples)



        return mu, sd


    def SE(self):
        """Create a squared exponential kernel

        Returns
        -------
        squared_exponential: function
            A function of two numpy.ndarrays of floats that computes the squared exponential
            kernel with given standard deviation and length scale.
        """

        def squared_exponential(x1,x2,sigma_f,l):
            """

            Emulates the model output given a set of input parameters.

            Args
            ----
            x1 : ndarray, shape (n, d)

            x2 : ndarray, shape (m, d)

            sigma_f : float
                The standard deviation of the kernel.

            l : float
                The length scale of the kernel.


            Returns
            -------
            K : ndarray, shape (n, m)
                Covariance matrix
            """

            if x1.ndim == 1:
                x1 = x1.reshape(-1, 1)
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)

            exp_term = np.exp( - (np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1)  - 2 * x1.dot(x2.T)) / (2*(l**2)))
            K = sigma_f**2 * exp_term
            return K
        
        return squared_exponential


    def linear_regression(self):

        """Performs linear regression

        Returns:
            coeff : ndarray, shape (order,)
                The regression coefficients.
            mu_ols : ndarray, shape (ntraining,)
                The regression fit to the training samples.
            var_ols : ndarray, shape (ntraining,)
                The variance of the fit.
        """

        X = self.design_matrix(self.input_train)

        def solve(A, y):
            coeff = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(y)
            return coeff.flatten()

        coeff = solve(X,self.output_train)

        mu_ols = np.dot(X, coeff)
        var_ols = np.dot((self.output_train - mu_ols).T, (self.output_train - mu_ols)) / len(mu_ols)

        return coeff, mu_ols, var_ols


    
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
            return X_d

class EigenvectorContinuation(Emulator):

    def emulate(self):
        pass
