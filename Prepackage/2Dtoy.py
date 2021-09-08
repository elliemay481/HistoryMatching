import numpy as np
import matplotlib.pyplot as plt
import random

# import internal files
import emulator
import kernels
import data
import historymatch
import plot

np.random.seed(12)

def true_model(x, y):
    return np.cos(np.sqrt(x**2 + 2*y**2)) + 1/3*x


# simulation parameters
Ntraining = 25          # number of training points
ndim = 2        # model dimensions
Nsamples = 1000    # number of test points
        
# define parameter space
x_bound = np.array([-3, 3]).reshape(1,-1)
y_bound = np.array([-3, 3]).reshape(1,-1)
input_bounds = np.concatenate((x_bound, y_bound), axis=0)

# for testing: true datapoints
true_x = 0.2
true_y = 0.2
var_exp = 0.001        # observational uncertainty variance
z = true_model(true_x, true_y) + np.random.normal(0,var_exp) # observed datapoint

# create squared-exponential kernel
ndim = 2              # no. of dimensions
sigma_cov = 0.2        # sqrt variance of covariance function
beta = 0         # prior expectation

kern = kernels.SE()

def history_match(parameter_bounds, sigma_cov, var_exp, beta, ndim, Ntraining, Nsamples, waves=1):
    
    # generate initial well spaced inputs for train and test sets
    input_train_all, input_test_all, output_train_all = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds, true_model)

    # evaluate true model over training inputs
    #output_train_all = np.zeros(Ntraining)
    #for i in range(Ntraining):
        #output_train_all[i] = true_model(input_train_all[i,0], input_train_all[i,1])

    # artificially add noise to observations
    output_train_all += np.random.normal(0,var_exp)
    
    # plot settings
    fig1, axes = plt.subplots(waves, 2, figsize=(12, 6*waves))
    ax_list = fig1.axes

    parameter_bounds_initial = parameter_bounds

    # only one initial implausible region 
    N_regions = 1
    
    for k in range(waves):

        print('Current wave: ' + str(k+1))

        input_train_list = []
        input_test_list = []
        output_train_list = []

        # iterate over nonimplausible regions
        for n in range(N_regions):

            print('Region: ' + str(n))

            if k == 0:
                input_train = input_train_all
                input_test = input_test_all
                output_train = output_train_all
            else:
                input_train = input_train_all[n]
                input_test = input_test_all[n]
                output_train = output_train_all[n]

            # build emulator over nonimplausible region
            GP = emulator.Gaussian_Process(input_train, input_test, output_train, sigma_cov, beta, kern)

            # optimise hyperparameters of emulator
            GP.optimise()
            # fit emulator using training points
            mu, cov, sd = GP.emulate()
            
            implaus = np.zeros(len(input_test))
            for i in range(len(input_test)):
                implaus[i] = historymatch.implausibility(mu[i], z, sd[i], 0, var_exp)
            
            # plot implausibilities
            ax1 = ax_list[2*k]
            ax2 = ax_list[2*k + 1]
            #ax3 = ax_list[3*k + 2]

            plot.implausibility_2D(input_test, implaus, parameter_bounds_initial, ax1, fig1, k, n)

            ax2.scatter(input_test[:,0],input_test[:,1], s=2, color = 'cornflowerblue', label='Implausible Points' if n == 0 else "")
            ax2.scatter(input_train[:,0], input_train[:,1], marker='x', color='black', label='Training Data' if n == 0 else "")
            
            # identify implausible region
            input_imp = np.concatenate((input_test, implaus.reshape(-1,1)), axis=1)
            # if region empty, skip
            nonimplausible = np.delete(input_imp, np.where(input_imp[:,2] > 3), axis=0)
            if nonimplausible.size == 0:
                continue

            # plot nonimplausible datapoints
            ax2.scatter(nonimplausible[:,0], nonimplausible[:,1], color='orange', s=2, label='Nonimplausible Points' if n == 0 else "")
            ax2.scatter(true_x, true_y, color='red', marker='x', label='Observed Data' if n == 0 else "")
            ax2.set_title('Remaining Non-Implausible Datapoints')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_xlim([-3,3])
            ax2.set_ylim([-3,3])
            ax2.legend(loc='lower left')

            # isolate implausible regions based on greatest y difference

            implaus_regions = historymatch.find_clusters_2D(nonimplausible, input_test, ndim, n_grid=10)


            
            # identify nonimplausible region boundaries and plot
            for i in range(len(implaus_regions)):
                group_i = implaus_regions[i]

                if group_i.shape[0] == 1:
                    xmin_i = group_i[:,0].min() - 0.01
                    xmax_i = group_i[:,0].max() + 0.01
                    ymin_i = group_i[:,1].min() - 0.01
                    ymax_i = group_i[:,1].max() + 0.01

                else:
                    xmin_i = group_i[:,0].min()
                    xmax_i = group_i[:,0].max()
                    ymin_i = group_i[:,1].min()
                    ymax_i = group_i[:,1].max()

                ratio = (ymax_i - ymin_i)/6
                Ntraining = int(np.ceil(ratio*25)) + 3
                Nsamples = int(np.ceil(ratio*1000)) + 100

                ax2.vlines(x=[xmin_i, xmax_i], ymin=ymin_i, ymax=ymax_i, linestyle = '--', linewidth=2, color='red')
                ax2.hlines(y=[ymin_i, ymax_i], xmin=xmin_i, xmax=xmax_i, linestyle = '--', linewidth=2, color='red')
                
                # find nonimplausible boundaries
                #x_bound = np.array([xmin_i, xmax_i]).reshape(1,-1)
                #y_bound = np.array([ymin_i, ymax_i]).reshape(1,-1)
                #parameter_bounds = np.concatenate((x_bound, y_bound), axis=0)


                parameter_bounds = data.locate_boundaries(group_i, ndim)
                # redefine nonimplausible space & generate new training points
                input_train_i, input_test_i, output_train_i = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds, true_model)
                # artificially add noise to observations
                output_train_i += np.random.normal(0,var_exp)

                input_train_list.append(np.array(input_train_i))
                input_test_list.append(np.array(input_test_i))
                output_train_list.append(np.array(output_train_i))
                
                
                
                
            
            '''
            n_grid = 20
            hist, xedges, yedges = np.histogram2d(nonimplausible[:,1], nonimplausible[:,0], bins=n_grid, range=[[-2, 2], [-2, 2]])

            im2 = ax3.contourf(np.linspace(-2,2,n_grid),np.linspace(-2,2,n_grid),hist/((100/20)**2),cmap='pink')
            cbar = fig.colorbar(im2, ax=ax3)
            im2.set_clim(0,1)'''
        
        input_train_all = input_train_list

        input_test_all = input_test_list
        output_train_all = output_train_list

        N_regions = len(input_train_all)

            #ax3.plot(x_test, implaus[:,0])
            #ax3.set_ylabel('Implausibility')
            #ax3.set_xlabel('x')
    fig1.savefig('implausibility_plots.png')
        

history_match(input_bounds, sigma_cov, var_exp, beta, 2, Ntraining, Nsamples, 3)


plt.show()