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

def true_model(x, y, a):
    return np.cos(np.sqrt(x**2 + 2*y**2 + a**2))
    #return np.sin(2*x)*np.cos(y)*np.sin(a)/5


# simulation parameters
Ntraining = 25          # number of training points
ndim = 2        # model dimensions
Nsamples = 1000    # number of test points

# define parameter space
x_bound = np.array([-2, 2]).reshape(1,-1)
y_bound = np.array([-2, 2]).reshape(1,-1)
a_bound = np.array([-2, 2]).reshape(1,-1)
input_bounds = np.concatenate((x_bound, y_bound, a_bound), axis=0)

# for testing: true datapoints
true_x = 0.1
true_y = 0.2
true_a = 0
var_exp = 0.001        # observational uncertainty variance
z = true_model(true_x, true_y, true_a) + np.random.normal(0,var_exp) # observed datapoint

# create squared-exponential kernel
sigma_cov = 0.2        # sqrt variance of covariance function
beta = 0         # prior expectation

kern = kernels.SE()


def history_match(parameter_bounds, sigma_cov, var_exp, beta, ndim, Ntraining, Nsamples, waves=1):
    
    # generate initial well spaced inputs for train and test sets
    input_train_all, input_test_all, output_train_all = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds, true_model)

    # artificially add noise to observations
    output_train_all += np.random.normal(0,var_exp)
    
    # plot settings
    fig1, axes = plt.subplots(waves, 2, figsize=(12, 6*waves))
    ax_list = fig1.axes

    parameter_bounds_initial = parameter_bounds
    
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
            #ax1 = fig1.add_subplot(111, projection='3d')
            ax2 = ax_list[2*k + 1]
            #ax3 = ax_list[3*k + 2]

            #ax1 = plt.axes(projection='3d')

            plot.implausibility_2D(input_test, implaus, parameter_bounds_initial, ax1, fig1, k, n)
            '''
            ax2.scatter(input_test[:,0],input_test[:,1], s=2, color = 'cornflowerblue', label='Implausible Points' if n == 0 else "")
            ax2.scatter(input_train[:,0], input_train[:,1], marker='x', color='black', label='Training Data' if n == 0 else "")
            '''
            # identify implausible region
            input_imp = np.concatenate((input_test, implaus.reshape(-1,1)), axis=1)
            # if region empty, skip
            nonimplausible = np.delete(input_imp, np.where(input_imp[:,-1] > 3), axis=0)
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

            implaus_regions = historymatch.find_clusters_3D(nonimplausible, input_test, ndim, parameter_bounds, n_grid=5)
            
            
            #print(implaus_regions)
            
            # identify nonimplausible region boundaries and plot
            for i in range(len(implaus_regions)):

                
                group_i = implaus_regions[i]
                #ax1.scatter(implaus_regions[i][:,0], implaus_regions[i][:,1], implaus_regions[i][:,2])

                parameter_bounds = np.empty((0,2))

                if group_i.shape[0] == 1:
                    for j in range(ndim):
                        min_i = group_i[:,j].min() - 0.01
                        max_i = group_i[:,j].max() + 0.01
                        parameter_bounds = np.concatenate((parameter_bounds, np.array([min_i, max_i]).reshape(1,-1)), axis=0)

                else:
                    for j in range(ndim):
                        min_i = group_i[:,j].min()
                        max_i = group_i[:,j].max()
                        parameter_bounds = np.concatenate((parameter_bounds, np.array([min_i, max_i]).reshape(1,-1)), axis=0)

                #ratio = (ymax_i - ymin_i)/6
                #Ntraining = int(np.ceil(ratio*25)) + 3
                #Nsamples = int(np.ceil(ratio*1000)) + 100

                #ax2.vlines(x=[xmin_i, xmax_i], ymin=ymin_i, ymax=ymax_i, linestyle = '--', linewidth=2, color='red')
                #ax2.hlines(y=[ymin_i, ymax_i], xmin=xmin_i, xmax=xmax_i, linestyle = '--', linewidth=2, color='red')
                
                # find nonimplausible boundaries
                #x_bound = np.array([xmin_i, xmax_i]).reshape(1,-1)
                #y_bound = np.array([ymin_i, ymax_i]).reshape(1,-1)
                #parameter_bounds = np.concatenate((x_bound, y_bound), axis=0)
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
    #fig1.savefig('implausibility_plots.png')
        

history_match(input_bounds, sigma_cov, var_exp, beta, 3, Ntraining, Nsamples, 3)


plt.show()