import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.optimize import basinhopping
from math import factorial

class Gaussian_Process:
    """
    Args:
        x_train (numpy.ndarray n x d) : Input training data
        x_test (numpy.ndarray m x d) : Input of well-spaced samples
        y_train (numpy.ndarray n x d) : Output training data
        sigma (float) : The standard deviation of the kernel
        l (float) : The length scale of the kernel.
        beta (float) : Prior expectation
        kernel (function)
        noise (bool)
        sigma_n (float) : standard devation of gaussian noise

    """
    def __init__(self, input_train, input_test, output_train, sigma_f=0.1, beta=0, l=0.5, noise=False, sigma_n=None):
        self.sigma_f = sigma_f
        self.kernel = self.SE()
        self.beta = beta
        self.l = l
        self.noise = noise
        self.sigma_n = 0.1
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train
        self.ndim = self.input_train.shape[1]




    def linear_regression(self):
        '''
        def design_matrix(x, order=2):
            N = self.ndim*(order) + 1
            ncombinations = factorial(self.ndim) / (2*factorial(self.ndim-2))
            X_d = np.zeros((len(x),N+self.ndim))
            X_d[:,0] = 1
            for i in range(order):
                for j in range(self.ndim):
                    X_d[:,1 + (i*self.ndim + j)] = x[:,j]**(i+1)
            prod = 0
            for i in range(self.ndim):
                for j in range(self.ndim):
                    if i < j:
                        X_d[:,N + prod] = x[:,i]*x[:,j]
                        prod += 1
            return X_d'''

        def design_matrix(x, order=1):
            N = self.ndim*(order) + 1
            X_d = np.zeros((len(x),N))
            X_d[:,0] = 1
            for i in range(order):
                for j in range(self.ndim):
                    X_d[:,1 + j] = x[:,j]**(i+1)
            return X_d

        X = design_matrix(self.input_train)
        X_test = design_matrix(self.input_test)

        def solve(A, y):
            coeff = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(y)
            return coeff.flatten()

        coeff = solve(X,self.output_train)
       
        mu_train = np.dot(X, coeff)

        mu_test = np.dot(X_test, coeff)
        #print((self.output_train - mu_train))
        var = np.dot((self.output_train - mu_train).T, (self.output_train - mu_train)) / len(mu_train)

        return mu_test, mu_train, var



    def emulate(self):
        """
            
        """

        # perform linear regression
        mu_ols, mu_train, var_ols = self.linear_regression()
        


        if self.noise == True:
            K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l) + (self.sigma_n**2)*np.eye(len(self.input_train))
        else:
            K_XX = self.kernel(self.input_train, self.input_train, self.sigma_f, self.l)

        K_XsX = self.kernel(self.input_test, self.input_train, self.sigma_f, self.l)
        K_XXs = self.kernel(self.input_train, self.input_test, self.sigma_f, self.l)
        K_XsXs = self.kernel(self.input_test, self.input_test, self.sigma_f, self.l)
        K_XX_inv = np.linalg.inv(K_XX)

        #print(K_XX)
        #print(np.linalg.det(K_XX))

        self.K_XX = K_XX
        self.K_XsX = K_XsX
        self.K_XXs = K_XXs
        self.K_XsXs = K_XsXs
        self.K_XX_inv = K_XX_inv

        
        #print(mu_ols)
        #print(self.K_XsX.dot(self.K_XX_inv).dot(self.output_train - self.beta))
        mu = mu_ols + self.K_XsX.dot(self.K_XX_inv).dot(self.output_train - mu_train)
        #mu = mu_ols
        cov = self.K_XsX.dot(self.K_XX_inv).dot(self.K_XXs)

        variance = var_ols + self.sigma_f**2 - np.abs(np.diag(cov))
        print(variance)
        #print(self.sigma_f**2)
        #print(np.abs(np.diag(cov)))
        
        sd = np.sqrt(variance)
        return mu, cov, sd
        



    def optimize(self):
        """
        Optimise GP hyperparameters
        *** Only for length scale now ***


        Returns:
        

        """

        '''
        def neg_log_marginal_likelihood(hyperparameters):
            
            sigma_f = self.sigma_f
            l = hyperparameters
            sigma_f = hyperparameters

            K_XX = self.kernel(self.input_train, self.input_train, sigma_f, l)

            K_XX_inv = np.linalg.inv(K_XX)

            if np.linalg.det(K_XX) == 0:
                return 1
            else:
                return 0.5 * ( (self.output_train.T).dot((K_XX_inv.dot(self.output_train)))
                         + np.log(np.linalg.det(K_XX)) + len(self.input_train)*np.log(2*np.pi) )
        
        '''

        def neg_log_marginal_likelihood(hyperparameters):

            l = hyperparameters[0]
            sigma_f = hyperparameters[1]

            K = self.kernel(self.input_train, self.input_train, sigma_f, l)
            y = self.output_train
            n = len(y)

            L = np.linalg.cholesky(K)
            alpha_0 = solve_triangular(L, y, lower=True)
            alpha = solve_triangular(L.T, alpha_0, lower=False)

            # check lower/upper triangles
            # check if y needs to be transposed


            #print(-0.5*(np.dot(y,alpha)) - np.sum(np.log(np.diagonal(L))) - 0.5*n*np.log(2*np.pi))
            return 0.5*(np.dot(y,alpha)) + np.sum(np.log(np.diagonal(L))) + 0.5*n*np.log(2*np.pi)
        
        bounds = [[1e-3, 1], [1e-3, 1]]
        #bounds = [[1e-3, 2]]
        for i in range(10):
            l_init = (bounds[0][1] - bounds[0][0]) * np.random.random() + bounds[0][0]
            s_init = (bounds[1][1] - bounds[1][0]) * np.random.random() + bounds[1][0]

            #print(l_init)
            #print(s_init)

            result = minimize(neg_log_marginal_likelihood, x0 = [l_init, s_init], bounds=bounds, method='L-BFGS-B')
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds":bounds}

        #result = basinhopping(neg_log_marginal_likelihood, x0=[1e-3], minimizer_kwargs=minimizer_kwargs, niter=200)
        #print(result.x)
        self.l, self.sigma_f = result.x
        #self.l = result.x
        




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
            K = sigma_f**2 * np.exp(- norm_sq / ((l**2)))
            return K
        
        return squared_exponential
