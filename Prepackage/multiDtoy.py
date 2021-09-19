import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec


# import internal files
import emulator
import kernels
import data
import historymatch
import plot

np.random.seed(12)

def model_eqn_1(x, y, a):
    return np.cos(np.sqrt(x**2 + 2*y**2 + 3*a**2))
    #return a + y + x

def model_eqn_2(x, y, a):
    #return np.sin(np.sqrt(x**2 + 4*y**2 + 2*a**2))
    return a**2 - y + x

true_model = [model_eqn_1, model_eqn_2]

# simulation parameters
Ntraining = 50          # number of training points
ndim = 3        # model dimensions
Nsamples = 10000    # number of test points

# define parameter space
x_bound = np.array([-2, 2]).reshape(1,-1)
y_bound = np.array([-2, 2]).reshape(1,-1)
a_bound = np.array([-2, 2]).reshape(1,-1)
input_bounds = np.concatenate((x_bound, y_bound, a_bound), axis=0)

# for testing: true datapoints
true_x = -0.1
true_y = 1.2
true_a = 0.3
true_parameters = [true_x, true_y, true_a]
var_exp = 0.001        # observational uncertainty variance
z_1 = model_eqn_1(true_x, true_y, true_a) + np.random.normal(0,var_exp) # observed datapoint
z_2 = model_eqn_2(true_x, true_y, true_a) + np.random.normal(0,var_exp) # observed datapoint

# create squared-exponential kernel
sigma_cov = 0.2        # sqrt variance of covariance function
beta = 0         # prior expectation

kern = kernels.SE()

        

historymatch.history_match(input_bounds, sigma_cov, var_exp, beta, Ntraining, Nsamples, zlist=[z_1, z_2], ndim=3, n_outputs=2, waves=4)


plt.show()