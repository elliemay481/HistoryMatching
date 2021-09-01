import numpy as np
import matplotlib.pyplot as plt

# import internal files
import emulator
import kernels
import data
import historymatch

def true_model(x, y):
    return np.cos(np.sqrt(x**2 + 2*y**2)) + 1/3*x


# simulation parameters
Ntraining = 25          # number of training points
ndim = 2        # model dimensions
Nsamples = 100    # number of test points
        
# define parameter space
x_bound = np.array([-2, 2]).reshape(1,-1)
y_bound = np.array([-2, 2]).reshape(1,-1)
input_bounds = np.concatenate((x_bound, y_bound), axis=0)

# for testing: true datapoints
true_x = 0.1
true_y = 0.2
z = true_model(true_x, true_y) # observed datapoint

# uncertainties
sigma_e = 0.1   # observational error

# create squared-exponential kernel
ndim = 2              # no. of dimensions
sigma_c = 0.1        # sqrt variance of covariance function
beta = 0         # prior expectation




kern = kernels.SE()

def history_match(parameter_bounds, sigma_c, sigma_n, beta, ndim, N_training_pts, N_test_pts, waves=1):
    
    input_train, input_test = data.prepare_data(ndim, Nsamples, Ntraining, input_bounds)

    output_train = np.zeros(Ntraining)
    for i in range(Ntraining):
        output_train[i] = true_model(input_train[i,0], input_train[i,1])


    N_test_pts = len(input_test)
    
    # plot settings
    fig, axes = plt.subplots(waves, 3, figsize=(16, 6*waves))
    ax_list = fig.axes
    
    
    for k in range(waves):
        
        print('Current wave: ' + str(k+1))
        
        xmin_0 = parameter_bounds[0,0]
        xmax_0 = parameter_bounds[0,1]
        ymin_0 = parameter_bounds[1,0]
        ymax_0 = parameter_bounds[1,1]

        GP = emulator.Gaussian_Process(input_train, input_test, output_train, sigma_c, beta, kern)
        
         # optimise hyperparameters of emulator
        GP.optimise()
        # fit emulator using training points
        mu, cov, sd = GP.emulate()
        
        implaus = np.zeros(N_test_pts)
        for i in range(N_test_pts):
            implaus[i] = historymatch.implausibility(mu[i], z, sd[i], 0, sigma_e**2)
        
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
        
        # find nonimplausible boundaries
        xmin = nonimplausible[:,0].min()
        xmax = nonimplausible[:,0].max()
        ymin = nonimplausible[:,1].min()
        ymax = nonimplausible[:,1].max()
        
        # redefine nonimplausible space
        
        input_test = nonimplausible[:,0:2]
        N_test_pts = len(input_test)

        
        # generate new training points in nonimplausible region
        
        boundaries = np.zeros((ndim, 2))
        for i in range(ndim):
            boundaries[i,0] = nonimplausible[:,i].min()
            boundaries[i,1] = nonimplausible[:,i].max()
            
        input_train = data.LHsampling(ndim, N_training_pts, boundaries)
        
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
        ax2.set_xlim([-2,2])
        ax2.set_ylim([-2,2])
        ax2.scatter(input_train[:,0], input_train[:,1], marker='x', color='black', label='Training Data')
        ax2.axhline(ymin, linestyle = '--', color='pink')
        ax2.axhline(ymax, linestyle = '--', color='pink')
        ax2.axvline(xmin, linestyle = '--', color='pink')
        ax2.axvline(xmax, linestyle = '--', color='pink')
        ax2.legend(loc='lower left')
        
            
        n_grid = 20
        hist, xedges, yedges = np.histogram2d(nonimplausible[:,1], nonimplausible[:,0], bins=n_grid, range=[[-2, 2], [-2, 2]])

        im2 = ax3.contourf(np.linspace(-2,2,n_grid),np.linspace(-2,2,n_grid),hist/((100/20)**2),cmap='pink')
        cbar = fig.colorbar(im2, ax=ax3)
        im2.set_clim(0,1)
        
        #ax3.plot(x_test, implaus[:,0])
        #ax3.set_ylabel('Implausibility')
        #ax3.set_xlabel('x')

history_match(input_bounds, sigma_c, 0, beta, 2, Ntraining, Nsamples, 3)

plt.show()