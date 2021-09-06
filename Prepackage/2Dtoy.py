import numpy as np
import matplotlib.pyplot as plt

# import internal files
import emulator
import kernels
import data
import historymatch

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
var_exp = 0.05        # observational uncertainty variance
z = true_model(true_x, true_y) + np.random.normal(0,var_exp) # observed datapoint

# create squared-exponential kernel
ndim = 2              # no. of dimensions
sigma_cov = 0.2        # sqrt variance of covariance function
beta = 0         # prior expectation



def group(data, stepsize=0.2):
    # check for groups spaced apart in y direction
    ygroups = np.split(np.sort(data[:,1]), np.where(np.diff(np.sort(data[:,1])) > stepsize)[0]+1)
    group_list = []
    # isolate groups within nonimplausible dataset
    for i in range(len(ygroups)):
        group_i = ygroups[i]
        temp, group_i_ind, temp_ind = np.intersect1d(data[:,1], group_i, return_indices=True)
        #groups_temp.append(data[group_i_ind])
        # check each group for spacing in x direction
        xgroups = np.split(np.sort(data[group_i_ind][:,0]), np.where(np.diff(np.sort(data[group_i_ind][:,0])) > stepsize)[0]+1)
        for j in range(len(xgroups)):
            temp, group_j_ind, temp_ind = np.intersect1d(data[:,0], xgroups[j], return_indices=True)
            group_list.append(data[group_j_ind])

    groups = np.array(group_list, dtype=object)
    return groups


kern = kernels.SE()

def history_match(parameter_bounds, sigma_cov, var_exp, beta, ndim, N_training_pts, N_test_pts, waves=1):
    
    # generate initial well spaced inputs for train and test sets
    input_train, input_test = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds)

    # evaluate true model over training inputs
    output_train = np.zeros(Ntraining)
    for i in range(Ntraining):
        output_train[i] = true_model(input_train[i,0], input_train[i,1])

    # artificially add noise to observations
    output_train += np.random.normal(0,var_exp)

    N_test_pts = len(input_test)
    
    # plot settings
    fig, axes = plt.subplots(waves, 3, figsize=(16, 6*waves))
    ax_list = fig.axes
    
    xmin_0 = parameter_bounds[0,0]
    xmax_0 = parameter_bounds[0,1]
    ymin_0 = parameter_bounds[1,0]
    ymax_0 = parameter_bounds[1,1]

    # only one initial implausible region 
    N_regions = 1
    
    for k in range(waves):

        print('Current wave: ' + str(k+1))

        if N_regions == 1:
            input_train = [input_train_n]
            input_test = [input_test_n]
            output_train

        # iterate over nonimplausible regions
        for n in N_regions:


        
        # build emulator over nonimplausible region
        GP = emulator.Gaussian_Process(input_train, input_test, output_train, sigma_cov, beta, kern)
        
         # optimise hyperparameters of emulator
        GP.optimise()
        # fit emulator using training points
        mu, cov, sd = GP.emulate()
        
        implaus = np.zeros(N_test_pts)
        for i in range(N_test_pts):
            implaus[i] = historymatch.implausibility(mu[i], z, sd[i], 0, var_exp)
        
        # plot implausibilities
        ax1 = ax_list[3*k]
        ax2 = ax_list[3*k + 1]
        ax3 = ax_list[3*k + 2]
        im =  ax1.tricontourf(input_test[:,0],input_test[:,1],implaus, levels=20, cmap='viridis_r')
        ax1.set_facecolor((68/255,1/255,84/255))
        ax1.set_xlim([xmin_0, xmax_0])
        ax1.set_ylim([ymin_0, ymax_0])
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Implausibility')
        im.set_clim(0,10)
        ax1.set_ylabel('y')
        ax1.set_xlabel('x')
        ax1.set_title('Wave ' + str(k+1) + ' Implausibility')
        ax1.scatter(true_x, true_y, color='red', marker='x', label='Observed Data')
        
        # identify implausible region
        input_imp = np.concatenate((input_test, implaus.reshape(-1,1)), axis=1)
        nonimplausible = np.delete(input_imp, np.where(input_imp[:,2] > 3), axis=0)

        # isolate implausible regions based on greatest y difference
        implaus_regions = group(nonimplausible)
        N_regions = len(implaus_regions)
        
        # identify nonimplausible region boundaries and plot
        for i in range(N_regions):
            group_i = group(nonimplausible)[i]

            xmin_i = group_i[:,0].min()
            xmax_i = group_i[:,0].max()
            ymin_i = group_i[:,1].min()
            ymax_i = group_i[:,1].max()
            ax2.vlines(x=[xmin_i, xmax_i], ymin=ymin_i, ymax=ymax_i, linestyle = '--', color='pink')
            ax2.hlines(y=[ymin_i, ymax_i], xmin=xmin_i, xmax=xmax_i, linestyle = '--', color='pink')

        print(implaus_regions.shape)

    
        # find nonimplausible boundaries
        x_bound = np.array([nonimplausible[:,0].min(), nonimplausible[:,0].max()]).reshape(1,-1)
        y_bound = np.array([nonimplausible[:,1].min(), nonimplausible[:,1].max()]).reshape(1,-1)
        
        xmin = parameter_bounds[0,0]
        xmax = parameter_bounds[0,1]
        ymin = parameter_bounds[1,0]
        ymax = parameter_bounds[1,1]

        # redefine nonimplausible space & generate new training points in nonimplausible region
        parameter_bounds = np.concatenate((x_bound, y_bound), axis=0)
        input_train, input_test = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds)

        # evaluate model over new training data
        output_train = np.zeros(N_training_pts)
        for i in range(N_training_pts):
            output_train[i] = true_model(input_train[i,0], input_train[i,1])
        
        # plot nonimplausible datapoints
        ax2.scatter(nonimplausible[:,0], nonimplausible[:,1], s=2)
        ax2.scatter(true_x, true_y, color='red', marker='x', label='Observed Data')
        ax2.set_title('Remaining Non-Implausible Datapoints')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_xlim([-3,3])
        ax2.set_ylim([-3,3])
        ax2.scatter(input_train[:,0], input_train[:,1], marker='x', color='black', label='Training Data')
        #ax2.axhline(ymin, linestyle = '--', color='pink')
        #ax2.axhline(ymax, linestyle = '--', color='pink')
        #ax2.axvline(xmin, linestyle = '--', color='pink')
        #ax2.axvline(xmax, linestyle = '--', color='pink')
        ax2.legend(loc='lower left')
        
            
        n_grid = 20
        hist, xedges, yedges = np.histogram2d(nonimplausible[:,1], nonimplausible[:,0], bins=n_grid, range=[[-2, 2], [-2, 2]])

        im2 = ax3.contourf(np.linspace(-2,2,n_grid),np.linspace(-2,2,n_grid),hist/((100/20)**2),cmap='pink')
        cbar = fig.colorbar(im2, ax=ax3)
        im2.set_clim(0,1)
        
        #ax3.plot(x_test, implaus[:,0])
        #ax3.set_ylabel('Implausibility')
        #ax3.set_xlabel('x')

history_match(input_bounds, sigma_cov, var_exp, beta, 2, Ntraining, Nsamples, 1)


plt.show()