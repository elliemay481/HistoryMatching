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


def history_match(parameter_bounds, sigma_cov, var_exp, beta, Ntraining, Nsamples, zlist, ndim, n_outputs, waves=1):
    
    # plot settings
    fig1 = plt.figure(figsize=(8, 6*waves))
    fig2, axes = plt.subplots(waves, 1, figsize=(8, 6*waves))
    gs0 = gridspec.GridSpec(waves, 1)
    ax_list = fig2.axes

    # find initial parameter volume
    initial_volume = 1
    for i in range(ndim):
        initial_volume = initial_volume * (parameter_bounds[i,1] - parameter_bounds[i,0])
    parameter_bounds_initial = parameter_bounds
    N_regions = 1

    Nsamples_0 = Nsamples
    Ntraining_0 = Ntraining
    for k in range(waves):

        print('Current wave: ' + str(k+1))

        input_train_list = []
        input_test_list = []
        output_train_list = []

        # plot settings
        gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[k], wspace=0.1, hspace=0.1)
        
        p1axes = np.empty((3,3), dtype=object)
        for i in range(3):
                for j in range(3):
                    p1axes[i,j] = fig1.add_subplot(gs00[i, j])

        # iterate over nonimplausible regions
        for n in range(N_regions):
            print('region: ' + str(n))

            # generate initial well spaced inputs for train and test sets
            if k == 0:
                input_train, input_test = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds)
                # save this input for optical depth plots - clean later *******
                input_test_0 = input_test
            else:
                input_train = input_train_all[n]
                input_test = input_test_all[n]

            # iterate over model outputs
            implaus_all = np.zeros((len(input_test), n_outputs))

            for output in range(n_outputs):


                if k == 0:
                    ax2 = plt.axes(projection='3d')
                    # evaluate true model over training inputs
                    output_train = np.zeros(Ntraining)
                    true_model_vec = np.vectorize(true_model[output])
                    output_train = true_model_vec(*input_train.T)
                    # artificially add noise to observations
                    output_train += np.random.normal(0,var_exp)
                    
                else:
                    output_train = output_train_all[n][output]

                # build emulator over nonimplausible region
                GP = emulator.Gaussian_Process(input_train, input_test, output_train, sigma_cov, beta, kern)
                # optimise hyperparameters of emulator
                GP.optimise()
                # fit emulator using training points
                mu, cov, sd = GP.emulate()

                # evaluate implausibility over parameter volume
                for i in range(len(input_test)):
                    implaus_all[i, output] = historymatch.implausibility(mu[i], zlist[output], sd[i], 0, var_exp)
                    
            # choose maximum implausibility
            max_I = np.argmax(implaus_all, axis=1)
            implaus = np.choose(max_I, implaus_all.T)

            # identify implausible region
            input_imp = np.concatenate((input_test, implaus.reshape(-1,1)), axis=1)
            # if region empty, skip
            nonimplausible = np.delete(input_imp, np.where(input_imp[:,-1] > 3), axis=0)
            if nonimplausible.size == 0:
                print('empty')
                continue

            # plot implausibilities and optical depth
            variable_names = ['x', 'y', 'z']
            for i in range(3):
                for j in range(3):
                    ax1 = p1axes[i,j]
                    variables = [i,j]
                    if i == j:
                        plot.optical_depth_1D(input_imp, 20, ax1, fig1, i, variable_names[i], parameter_bounds_initial, input_test_0)
                    elif i > j:
                        plot.implausibility(input_imp, parameter_bounds_initial, ax1, fig1, k, n, variables, [variable_names[i], variable_names[j]])
                        #ax1.scatter(true_parameters[j], true_parameters[i], color='red', marker='x', label='Observed Data' if n == 0 else "")
                    else:
                        plot.optical_depth_2D(input_imp, parameter_bounds_initial, input_test_0, ax1, fig1, k, n, variables, [variable_names[i], variable_names[j]])

            # isolate implausible regions based on greatest y difference
            implaus_regions = historymatch.find_clusters_3D(nonimplausible, input_test, ndim, parameter_bounds, n_grid=5)
            implaus_volumes = []
            implaus_N_pts = []
            # identify nonimplausible region boundaries and plot
            for i in range(len(implaus_regions)):

                group_i = implaus_regions[i]

                # plot first wave regions to visually check results
                if k == 0:
                    
                    ax2.set_title('Wave 1 Nonimplausible Regions')
                    ax2.scatter(implaus_regions[i][:,0], implaus_regions[i][:,1], implaus_regions[i][:,2])
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                    ax2.set_zlabel('z')
                    ax2.set_xlim([parameter_bounds_initial[0,0],parameter_bounds_initial[0,1]])
                    ax2.set_ylim([parameter_bounds_initial[1,0],parameter_bounds_initial[1,1]])
                

                # find nonimplausible boundaries
                parameter_bounds = np.empty((0,2))
                volume = 1
                if group_i.shape[0] == 1:
                    for j in range(ndim):
                        min_i = group_i[:,j].min() - 0.01
                        max_i = group_i[:,j].max() + 0.01
                        parameter_bounds = np.concatenate((parameter_bounds, np.array([min_i, max_i]).reshape(1,-1)), axis=0)
                        volume = volume * (max_i - min_i)

                else:
                    for j in range(ndim):
                        min_i = group_i[:,j].min()
                        max_i = group_i[:,j].max()
                        parameter_bounds = np.concatenate((parameter_bounds, np.array([min_i, max_i]).reshape(1,-1)), axis=0)
                        volume = volume * (max_i - min_i)

                # rescale number of points to sample in parameter space
                implaus_volumes.append(volume)
                
            total_volume = sum(implaus_volumes)

            for i in range(len(implaus_regions)):

                Ntraining = int(np.ceil((implaus_volumes[i]/total_volume)*Ntraining))
                Nsamples = int(np.ceil((implaus_volumes[i]/total_volume)*Nsamples))

                # redefine nonimplausible space & generate new training points
                input_train_i, input_test_i = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds)

                # evaluate true model over training inputs
                output_train_region = []
                for output in range(n_outputs):
                    output_train_i = np.zeros(len(input_train_i))
                    true_model_vec = np.vectorize(true_model[output])
                    output_train_i = true_model_vec(*input_train_i.T)
                    # artificially add noise to observations
                    output_train_i += np.random.normal(0,var_exp)
                    output_train_region.append(np.array(output_train_i))

                input_train_list.append(input_train_i)
                input_test_list.append(input_test_i)
                output_train_list.append(output_train_region)
        
        input_train_all = input_train_list

        input_test_all = input_test_list
        output_train_all = output_train_list

        N_regions = len(input_train_all)

    fig1.savefig('implausibility_plots_3D.png')
    fig2.savefig('test_plots_3D.png')
        

history_match(input_bounds, sigma_cov, var_exp, beta, Ntraining, Nsamples, zlist=[z_1, z_2], ndim=3, n_outputs=2, waves=4)


plt.show()