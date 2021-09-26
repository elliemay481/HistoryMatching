import numpy as np
import emulator
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
from scipy.stats import norm, uniform
from scipy import stats
import nestle

import plot
import data


# define implausibility measure
def implausibility(E, z, var, var_md, var_obs):
    
    # E - emulator expectation
    # z - observational data
    # var_em - emulator_uncertainty
    # var_md - model discrepency error
    # var_obs - observational error
    
    return np.sqrt( ( E - z )**2  /  ( var + var_md + var_obs ) )


def find_bounding_ellipse(points, volume=None):

    ells = nestle.bounding_ellipsoid(points, volume)
    cov = np.linalg.inv(ells.a)
    return ells.ctr, cov, ells




def sample_volume(ndim, covariance, nsamples, ellipse):

    C = covariance
    Cinv = np.linalg.inv(C)

    lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(np.linalg.det(C)))

    def prior(x):
        return 2. * (2. * x - 1.)

    def loglikelihood(x):
        """Multivariate normal log-likelihood."""
        return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

    res = nestle.sample(loglikelihood, prior, ndim=ndim, npoints=nsamples, method='single')

    return ellipse.samples(nsamples)

    #return res.samples


def history_match(true_model, obs_data, xvals, kernel, ndim, Nsamples, Ntraining, parameter_bounds, var_method, var_exp, sigma_cov, beta, theta, sigma_n, H, true_thetas, noise=False, waves=1):
    
    '''
        xvals: (1xNx) array of independent variable values
        obs_data = (Nfunc x Nx) array of function outputs
    '''

    Nx = len(xvals)      # number of independent variable, x, values
    Nfunc = len(true_model)

    fig, axes = plt.subplots(waves, 1, figsize=(8, 6*waves))
    gs0 = gridspec.GridSpec(waves, 1)
    
    ax_list = fig.axes

    color_list = ['plum', 'mediumaquamarine']

    parameter_bounds_initial = parameter_bounds


    for k in range(waves):

        print('Current wave: ' + str(k+1))

        ax_list[k].axis('off')
        gs00 = gridspec.GridSpecFromSubplotSpec(ndim, ndim, subplot_spec=gs0[k], wspace=0.1, hspace=0.1)
        p1axes = np.empty((ndim,ndim), dtype=object)
        for i in range(ndim):
            for j in range(ndim):
                p1axes[i,j] = fig.add_subplot(gs00[i, j])

        # for each x value, need a separate emulator. can use same train and test regions
        # for parameters though
        # generate initial well spaced inputs for train and test sets
        if k == 0:
            theta_train, theta_test = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds)
        # each model will have Nx outputs, so will have Nx*number of models outputs in total
        implausibility_all = np.zeros((Nsamples, Nx*Nfunc))

        for m in range(len(true_model)):
            true_function = true_model[m]

            # then iterate over x values
            for x in range(len(xvals)):

                xval = xvals[x]

                # generate training outputs
                z_train = true_function(xval, *theta_train.T)
                # artificially add noise to observations
                #z_train += np.random.normal(0,var_exp, Ntraining)

                # **** for testing without emulator *****
                #mu = true_function(xval, *theta_test.T)
                #sd = np.zeros(len(theta_test))

                # build emulator over nonimplausible region
                GP = emulator.Gaussian_Process(theta_train, theta_test, z_train, sigma_cov, beta, theta, kernel, noise, sigma_n)
                # optimise hyperparameters of emulator
                #GP.optimise()
                # fit emulator using training points
                mu, cov, sd = GP.emulate()

                # evaluate implausibility over parameter volume
                for i in range(len(theta_test)):
                    implausibility_all[i, x+(Nx*m)] = implausibility(mu[i], obs_data[m, x], sd[i], var_method, var_exp)


        # choose maximum (or second maximum) implausibility
        # get index of maximum implaus for all outputs
        max_I = np.argmax(implausibility_all, axis=1)
        # get index of second highest maximum implaus for all outputs
        max2_I = implausibility_all.argsort()[:,-2]
        implausibilities = implausibility_all[range(len(max2_I)), max2_I]

        # identify nonimplausible region
        samples_implaus = np.concatenate((theta_test, implausibilities.reshape(-1,1)), axis=1)
        nonimplausible = np.delete(samples_implaus, np.where(samples_implaus[:,-1] > 3), axis=0)

        ctr, cov, ell = find_bounding_ellipse(nonimplausible[:,:ndim], 0)
        
        # plot implausibilities and optical depth
        variable_names = [r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$']
        for i in range(ndim):
            for j in range(ndim):
                ax = p1axes[i,j]
                variables = [i,j]
                if i == j:
                    if i == 0:
                        ax.set_ylabel(variable_names[i])
                    elif i == ndim-1:
                        ax.set_xlabel(variable_names[i])
                    plot.optical_depth_1D(samples_implaus, 20, ax, fig, i, variable_names[i], parameter_bounds_initial)
                    #ax_right = ax.twinx()
                    #theta_vals = np.linspace(parameter_bounds_initial[i,0], parameter_bounds_initial[i,1], 100)
                    #for m in range(len(true_model)):
                        #ax_right.plot(theta_vals, stats.norm.pdf(theta_vals, true_thetas[m][i], np.sqrt(H[m][i,i])), color=color_list[m])
                    '''
                    if i < ndim-1:
                        if i == 0:
                            ax.set_ylabel(variable_names[i])
                        ax.scatter(nonimplausible[:,i], nonimplausible[:,i+1], s=1)
                        ax.set_xlim([parameter_bounds_initial[i,0], parameter_bounds_initial[i,1]])
                        ax.set_ylim([parameter_bounds_initial[i+1,0], parameter_bounds_initial[i+1,1]])
                        covi = np.array([[cov[i,i], cov[i,i+1]],[cov[i+1,i], cov[i+1,i+1]]])
                        plot.get_cov_ellipse(covi, [ctr[i],ctr[i+1]], 1, ax, color='red')
                    else:
                        ax.scatter(nonimplausible[:,0], nonimplausible[:,i], s=1)
                        ax.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
                        ax.set_ylim([parameter_bounds_initial[i,0], parameter_bounds_initial[i,1]])
                        covi = np.array([[cov[0,0], cov[0,i]],[cov[i,0], cov[i,i]]])
                        plot.get_cov_ellipse(covi, [ctr[0],ctr[i]], 1, ax, color='red')
                        ax.set_xlabel(variable_names[i])'''
                    

                elif i > j:
                    #plot.implausibility_2D(samples_implaus, parameter_bounds_initial, ax1, fig1, k, n, variables, [variable_names[j], variable_names[i]])
                    plot.implausibility(samples_implaus, parameter_bounds_initial, ax , fig, k, 0, [j,i], 
                            [variable_names[j], variable_names[i]], bins=30)
                    for m in range(len(true_model)):
                        ax.scatter(true_thetas[m][j],true_thetas[m][i], color='red', marker='x')
                #ax1.scatter(true_parameters[j], true_parameters[i], color='red', marker='x', label='Observed Data' if n == 0 else "")
                else:
                    plot.optical_depth_2D(samples_implaus, parameter_bounds_initial, ax, fig, k, [j,i], [variable_names[j], variable_names[i]])
                    #cov_matrix2 = np.array([[cov[i,i], cov[i,j]],[cov[j,i], cov[j,j]]])
                    #plot.get_cov_ellipse(cov_matrix2, [ctr[i],ctr[j]], 1, ax, 'red')
                    #for m in range(len(true_model)):
                        #cov_matrix = np.array([[H[m][i,i], H[m][i,j]],[H[m][j,i], H[m][j,j]]])
                        #plot.get_cov_ellipse(cov_matrix, [true_thetas[m][i],true_thetas[m][j]], 3, ax, color_list[m])
                        #ax.set_xlim([parameter_bounds_initial[i,0], parameter_bounds_initial[i,1]])
                        #ax.set_ylim([parameter_bounds_initial[j,0], parameter_bounds_initial[j,1]])
        
        # sample new implausible volume
        theta_train = sample_volume(ndim, cov, nsamples=Ntraining, ellipse=ell)
        theta_test = sample_volume(ndim, cov, nsamples=Nsamples, ellipse=ell)

        # if no points left in implausible region, end
        if nonimplausible.size == 0:
                print('Nonimplausible region empty')
                return None

    return ctr, cov, nonimplausible, ell

            
