import numpy as np
from pyDOE import lhs
from scipy.stats import norm
import utils
import sample

def LHsampling(ndim, Ntraining, limits):
    
    '''
    Args:
    ndim : Number of dimensions
    Ntraining : Number of training points
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns: (d x M) array of training points
    
    '''
    
    # generate sample points
    input_train = lhs(ndim, samples=Ntraining, criterion='center')
    print(limits.shape)
    
    # adjust sample points for parameter ranges
    for i in range(ndim):
        input_train[:,i] = input_train[:,i]*(limits[i,1]-limits[i,0]) + limits[i,0]

    
    return input_train




def hypercube_sample(ndim, Nsamples, Ntraining, limits):

    '''
    Args:
    ndim : Number of dimensions
    Ntraining : Number of training points
    Nsamples : Number of points along parameter space axes
    Limits: (n x d) array of upper and lower boundaries of parameter region
    
    Returns:
        (Ntraining x ndim) array of training points
        (Nsamples x ndim) array of well spaced samples in parameter space
    
    '''

    # generate sample points on Latin Hypercube
    input_test = LHsampling(ndim, Nsamples, limits)
    input_train = LHsampling(ndim, Ntraining, limits)

    return input_train, input_test


def ellipsoid_sample(ndim, Nsamples, Ntraining, mean, covariance):

    '''
    Args:
    ndim : Number of dimensions
    Nsamples : Number of points along parameter space axes
    Ntraining : Number of training points
    mean : ndim length array of mean parameter values
    covariance : (ndim x ndim) array
    
    Returns:
        (Ntraining x ndim) array of training points
        (Nsamples x ndim) array of well spaced samples in parameter space
    
    '''
    # generate sample points
    u_train = lhs(ndim, samples=Ntraining, criterion='center')
    u_test = lhs(ndim, samples=Nsamples, criterion='center')

    # normally distribute sample points
    for i in range(ndim):
        u_train[:,i] = norm(loc=0, scale=1).ppf(u_train[:,i])
        u_test[:,i] = norm(loc=0, scale=1).ppf(u_test[:,i])
    
    # Add pertubation to covariance
    epsilon = 1e-9
    K = covariance + epsilon*np.identity(ndim)

    

    # Calculate the Cholesky decomposition
    L = np.linalg.cholesky(K)

    # Compute multivariate gaussian distributed samples
    input_train = mean.reshape(ndim, 1) + np.dot(L, u_train.reshape(ndim, Ntraining))
    input_test = mean.reshape(ndim, 1) + np.dot(L, u_test.reshape(ndim, Nsamples))

    return input_train.T, input_test.T


def rotated_hypercube_samples(ndim, nonimp_vol, Nsamples, Ntraining):

    # change this!!
    # find angle of samples
    eigvals, eigvecs = np.linalg.eigh(np.cov(nonimp_vol.T))
    #order = eigvals.argsort()[::-1]
    #eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    #vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]

    #theta = -np.pi/2 + np.arctan2(vy, vx)
    #theta2 = np.pi/2 -np.arctan2(vy, vx)

    #c, s = np.cos(theta), np.sin(theta)
    #R = np.array([[c,-s],[s,c]])

    R = np.concatenate((eigvecs[0].reshape(-1,1),eigvecs[1].reshape(-1,1),eigvecs[2].reshape(-1,1)),axis=1)

    #c_r, s_r = np.cos(theta2), np.sin(theta2)
    #R_reverse = np.array([[c_r,-s_r],[s_r,c_r]])

    nonimp_str = np.zeros((len(nonimp_vol),ndim))
    for i in range(len(nonimp_vol)):
        #nonimp_str[i] = np.dot(R_reverse, nonimp_vol[i])
        nonimp_str[i] = np.dot(R.T, nonimp_vol[i])

    bounds = utils.locate_boundaries(nonimp_str, ndim)

    print('x :' + str(bounds[0,1]-bounds[0,0]))
    print('y :' + str(bounds[1,1]-bounds[1,0]))
    print('z :' + str(bounds[2,1]-bounds[2,0]))


    samples_test = sample.LHsampling(ndim, Nsamples, bounds)
    samples_train = sample.LHsampling(ndim, Ntraining, bounds)

    input_test = np.zeros((len(samples_test),ndim))
    input_train = np.zeros((len(samples_train),ndim))

    for i in range(len(samples_test)):
        input_test[i] = np.dot(R, samples_test[i])

    for i in range(len(samples_train)):
        input_train[i] = np.dot(R, samples_train[i])

    return input_train, input_test