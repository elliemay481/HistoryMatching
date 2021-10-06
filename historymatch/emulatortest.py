import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.optimize import basinhopping

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
    def __init__(self, input_train, input_test, output_train, sigma_f=0.01, beta=0, l=0.5, noise=False, sigma_n=None):
        self.sigma_f = sigma_f
        self.kernel = self.SE()
        self.beta = beta
        self.l = l
        self.noise = noise
        self.sigma_n = 0.1
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train




    def emulate(self):
        """
            
        """

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
            print(result.x)
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds":bounds}

        #result = basinhopping(neg_log_marginal_likelihood, x0=[1e-3], minimizer_kwargs=minimizer_kwargs, niter=200)
        print(result.x)
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
