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

random.seed(1)

# define implausibility measure
def implausibility(E, z, var, var_md, var_obs):
    
    # E - emulator expectation
    # z - observational data
    # var_em - emulator_uncertainty
    # var_md - model discrepency error
    # var_obs - observational error
    
    return np.sqrt( ( E - z )**2  /  ( var + var_md + var_obs ) )

def wave(input_test, z, sigma_e, GP, ax):

    # optimise hyperparameters of emulator
    GP.optimise()
    # fit emulator using training points
    mu, cov, sd = GP.emulate()
    # evaluate implausibility
    implaus = np.zeros(len(input_test))
    for i in range(len(input_test)):
        implaus[i] = implausibility(mu[i], z, sd[i], 0, sigma_e**2)

    im = ax.tricontourf(input_test[:,0],input_test[:,1],implaus, levels=30, cmap='viridis_r')
    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([xmin_0, xmax_0])
    ax.set_ylim([ymin_0, ymax_0])
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Implausibility')
    im.set_clim(0,10)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('Wave ' + str(k+1) + ' Implausibility')
    return im


def evaluate_cell(ax2, cell, n_grid, parameter_bounds, points, ndim):

    cell_volume = 1

    previous_indices = np.array([])

    cell_limits = np.zeros((ndim, 2))

    for i in range(ndim):
        i_range = parameter_bounds[i,1] - parameter_bounds[i,0]
        cell_i = i_range / n_grid
        cell_volume = cell_volume * cell_i
        min_i = parameter_bounds[i,0] + cell_i*(cell[i]%n_grid)
        max_i = parameter_bounds[i,0] + cell_i*(cell[i]%n_grid + 1)
        cell_limits[i,0] = min_i
        cell_limits[i,1] = max_i

        indices = np.where(np.logical_and(points[:,i]>=min_i, points[:,i]<=max_i))

        if i == 0:
            previous_indices = indices
        else:
            common_indices = np.intersect1d(indices, previous_indices)
            previous_indices = common_indices

    density = len(common_indices) / cell_volume

    return density, cell_limits

def find_bounding_ellipse(points, volume=None):
    ells = nestle.bounding_ellipsoid(points, volume)
    ctrlist = []
    covlist = []
    #for i in range(len(ells)):
        #cov = np.linalg.inv(ells[i].a)
        #covlist.append(cov)
        #ctrlist.append(ells[i].ctr)

    cov = np.linalg.inv(ells.a)
    return ells.ctr, cov, ells


def find_clusters_3D(ax2, dataset, input_test, ndim, parameter_bounds, n_grid = 1, threshold=1):

    # define parameter space
    parameter_bounds = data.locate_boundaries(input_test, ndim)

    cluster_bounds = []

    points = np.delete(dataset, -1, axis=1)

    # divide data into grid of cells
    cellsides = [np.arange(0,n_grid,1) for dim in range(ndim)]
    all_cells = np.array(np.meshgrid(*cellsides)).T.reshape(-1,ndim)
    N_cells = n_grid**ndim
    traversed_cells = []
 
    while len(traversed_cells) < N_cells:

        cell_bounds = []
        # randomly select cell
        cell_index = np.random.randint(0, N_cells)
        cell = all_cells[cell_index]
        # check if in list. if so, generate another
        while cell_index in traversed_cells:
            cell_index = np.random.randint(0, N_cells)
            cell = all_cells[cell_index]
        # calculate density of cell
        traversed_cells.append(cell_index)
        density, cell_limits =  evaluate_cell(ax2, cell, n_grid, parameter_bounds, points, ndim) 
        # check if density is greater than threshold
        if density > threshold:
            # mark cell as new cluster
            #cluster = np.append(cluster, points[cell_pts], axis=0)
            cell_bounds.append(cell_limits)
            # calculate density of neighbouring cells
            neighbours = []
            for j in range(ndim):
                n_1 = [val for val in cell]
                n_2 = [val for val in cell]
                n_1[j] += 1
                n_2[j] += -1
                n1_index = np.where((all_cells == n_1).all(axis=1))[0]
                n2_index = np.where((all_cells == n_2).all(axis=1))[0]
                if (n1_index not in traversed_cells) & (n1_index.size != 0):
                    neighbours.append(n_1)
                if (n2_index not in traversed_cells) & (n2_index.size != 0):
                    neighbours.append(n_2)
            while len(neighbours) != 0:
                for neighbour in neighbours:
                    neigh_index = np.where((all_cells == neighbour).all(axis=1))[0]
                    # if cell not checked already, add to traversed list
                    if neigh_index not in traversed_cells:
                        traversed_cells.append(neigh_index)
                        # check density of neighbouring cell
                        density_neigh, neigh_cell_limits = evaluate_cell(ax2, neighbour, n_grid, parameter_bounds, points, ndim)
                        if density_neigh > threshold:
                            # mark cell as part of cluster
                            cell_bounds.append(neigh_cell_limits)
                            # check cell neighbours
                            for j in range(ndim):
                                n_1 = [val for val in neighbour]
                                n_2 = [val for val in neighbour]
                                n_1[j] += 1
                                n_2[j] += -1
                                n1_index = np.where((all_cells == n_1).all(axis=1))[0]
                                n2_index = np.where((all_cells == n_2).all(axis=1))[0]
                                if (n1_index not in traversed_cells) & (n1_index.size != 0):
                                    neighbours.append(n_1)
                                if (n2_index not in traversed_cells) & (n2_index.size != 0):
                                    neighbours.append(n_2)
                            neighbours.remove(neighbour)
                        else:
                            neighbours.remove(neighbour)
                    else:
                        neighbours.remove(neighbour)
            else:
                # no more neighbours, mark as complete cluster
                cluster_bounds.append(cell_bounds)
                
        else:
            pass
    return cluster_bounds

'''def history_match_grid(kernel, true_model, parameter_bounds, sigma_cov, var_exp, var_method, beta, theta, Ntraining, Nsamples, zlist, ndim, n_outputs, waves=1, xvals=None, yvals=None, idx_joint=None, noise=False, sigma_n=None, plotOD=False):
    
    # plot settings
    if plotOD == True:
        fig2, axes = plt.subplots(waves, 1, figsize=(8, 6*waves))
        gs0 = gridspec.GridSpec(waves, 1)
        ax_list = fig2.axes
    else:
         # plot settings
        fig1, axes = plt.subplots(waves, 3, figsize=(15, 6*waves))
        ax_list = fig1.axes


    # find initial parameter volume
    initial_volume = 1
    for i in range(ndim):
        initial_volume = initial_volume * (parameter_bounds[i,1] - parameter_bounds[i,0])
    
    # save initial parameters for rescaling later
    parameter_bounds_initial = parameter_bounds
    N_regions = 1
    Nsamples_0 = Nsamples
    Ntraining_0 = Ntraining

    # store results for 1d plotting
    emulator_mean = []
    emulator_sd = []
    wave_samples = []
    wave_implaus = []


    for k in range(waves):

        print('Current wave: ' + str(k+1))

        if plotOD == False:
            # plot settings
            ax1 = ax_list[3*k]
            ax2 = ax_list[3*k + 1]
            ax3 = ax_list[3*k + 2]
        else:
            # plot settings
            gs00 = gridspec.GridSpecFromSubplotSpec(ndim, ndim, subplot_spec=gs0[k], wspace=0.1, hspace=0.1)
            p1axes = np.empty((ndim,ndim), dtype=object)
            for i in range(ndim):
                for j in range(ndim):
                    p1axes[i,j] = fig1.add_subplot(gs00[i, j])

        input_train_list = []
        input_test_list = []
        output_train_list = []
        all_implaus_regions = []
        parameter_bounds_list = []

    
        # iterate over nonimplausible regions
        for n in range(N_regions):
            print('region: ' + str(n))

            # generate initial well spaced inputs for train and test sets
            if k == 0:
                input_train, input_test = data.prepare_data(ndim, Nsamples, Ntraining, [parameter_bounds], ncells=1)
            else:
                input_train = input_train_all[n]
                input_test = input_test_all[n]

            # iterate over model outputs
            implausibility_all = np.zeros((len(input_test), n_outputs))

            for output in range(n_outputs):

                if k == 0:
                    # evaluate true model over training inputs
                    output_train = np.zeros(Ntraining)
                    true_model_vec = np.vectorize(true_model[output])
                    output_train = true_model_vec(*input_train.T)
                    # artificially add noise to observations
                    output_train += np.random.normal(0,var_exp)
                    
                else:
                    output_train = output_train_all[n][output]

                # build emulator over nonimplausible region
                GP = emulator.Gaussian_Process(input_train, input_test, output_train, sigma_cov, beta, theta, kernel, noise, sigma_n)
                # optimise hyperparameters of emulator
                #GP.optimise()
                # fit emulator using training points
                mu, cov, sd = GP.emulate()

                # save 1st output, f(x,y) for 1d plotting
                if output == 1:
                    emulator_mean.append(mu)
                    emulator_sd.append(sd)
                    wave_samples.append(input_test)

                # evaluate implausibility over parameter volume
                for i in range(len(input_test)):
                    implausibility_all[i, output] = implausibility(mu[i], zlist[output], sd[i], var_method, var_exp)
                    
            # choose maximum implausibility
            max_I = np.argmax(implausibility_all, axis=1)
            implausibilities = np.choose(max_I, implausibility_all.T)

            # save for plotting

            wave_implaus.append(implausibilities)

            # identify implausible region
            samples_implaus = np.concatenate((input_test, implausibilities.reshape(-1,1)), axis=1)
            nonimplausible = np.delete(samples_implaus, np.where(samples_implaus[:,-1] > 3), axis=0)

            # if region empty, skip
            if nonimplausible.size == 0:
                print('empty')
                continue

            # plot implausibilities and optical depth
            if plotOD == True:
                variable_names = ['x', 'y', 'z']
                for i in range(ndim):
                    for j in range(ndim):
                        ax1 = p1axes[i,j]
                        variables = [i,j]
                        if i == j:
                            plot.optical_depth_1D(samples_implaus, 20, ax1, fig1, i, variable_names[i], parameter_bounds_initial)
                        elif i > j:
                            #plot.implausibility_2D(samples_implaus, parameter_bounds_initial, ax1, fig1, k, n, variables, [variable_names[j], variable_names[i]])
                            plot.implausibility(samples_implaus, parameter_bounds_initial, ax1 , fig1, k, n, [j,i], 
                                    [variable_names[j], variable_names[i]], bins=30)
                            #ax1.scatter(true_parameters[j], true_parameters[i], color='red', marker='x', label='Observed Data' if n == 0 else "")
                        else:
                            plot.optical_depth_2D(samples_implaus, parameter_bounds_initial, ax1, fig1, k, n, variables, [variable_names[i], variable_names[j]])
            else:
                ax2.scatter(nonimplausible[:,0], nonimplausible[:,1], s=2, alpha=0.3)
                # plot implausibilities
                variable_names = ['x', 'y', 'z']
                plot.implausibility(samples_implaus, parameter_bounds_initial, ax1 , fig1, k, n, [0,1], 
                                   [variable_names[0], variable_names[1]], bins=30)


            # isolate implausible regions based on greatest y difference
            clusters = find_clusters_3D(ax2, nonimplausible, input_test, ndim, parameter_bounds, n_grid=15)
            
            # find cluster sizes in comparison to total implausible volume
            total_volume = 0
            for cluster in range(len(clusters)):
                all_implaus_regions.append(clusters[cluster])
                total_volume += len(clusters[cluster])
                
                if plotOD == False:
                    # plot cells to check
                    for cell in range(len(clusters[cluster])):
                        ax2.fill_between([clusters[cluster][cell][0,0], clusters[cluster][cell][0,1]], clusters[cluster][cell][1,0], clusters[cluster][cell][1,1], color='red', alpha=0.2)



        # identify nonimplausible region boundaries and plot
        for i in range(len(all_implaus_regions)):

            region = all_implaus_regions[i]

            # rescale number of samples based on cluster size
            Ntraining = int(np.ceil((len(region)/total_volume)*Ntraining_0))
            Nsamples = int(np.ceil((len(region)/total_volume)*Nsamples_0))

                
            # redefine nonimplausible space & generate new training points
            input_train_i, input_test_i = data.prepare_data(ndim, Nsamples, Ntraining, region, len(region))


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

        if plotOD == False:
            # plot true 3sigma region
            #ax3.scatter(xvals[idx_joint[:,1]], yvals[idx_joint[:,0]], color='fuchsia', alpha=0.3, label=r'Joint $3\sigma$ region')
            ax2.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
            ax2.set_ylim([parameter_bounds_initial[1,0], parameter_bounds_initial[1,1]])
            ax3.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
            ax3.set_ylim([parameter_bounds_initial[1,0], parameter_bounds_initial[1,1]])
                
            #ax1.set_title('Wave {} Implausibility'.format(str(k+1)))
            #ax3.set_title(r'True $3\sigma$ Region')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')

    #fig1.savefig('implausibility_plots_3D.png')
    #fig2.savefig('test_plots_3D.png')
    return emulator_mean, emulator_sd, wave_samples, wave_implaus'''


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
        theta_train, theta_test = data.prepare_data(ndim, Nsamples, Ntraining, parameter_bounds)
        
        # each model will have Nx outputs, so will have Nx*number of models outputs in total
        implausibility_all = np.zeros((Nsamples, Nx*Nfunc))

        for m in range(len(true_model)):
            true_function = true_model[m]

            # then iterate over x values
            for x in range(len(xvals)):

                # each x value has a different output. evaluate true model with this x value.
                # in the real case, there will be a set of data inputted into package. so take
                # this as function input instead. will be a (n x Nx) array with n = number of 
                # functions and Nx = number of xvals/outputs per function. but how do we know
                # the theta that this is for? have to be able to generate new output train
                # for each wave. so must be inside?
                # but will have true values inputted, this is the obs data!

                z_train = true_function(x, *theta_train.T)
                # artificially add noise to observations
                # should this be an array of noise? yes
                z_train += np.random.normal(0,var_exp, Ntraining)

                # build emulator over nonimplausible region
                GP = emulator.Gaussian_Process(theta_train, theta_test, z_train, sigma_cov, beta, theta, kernel, noise, sigma_n)
                # optimise hyperparameters of emulator
                GP.optimise()
                # fit emulator using training points
                mu, cov, sd = GP.emulate()

                # evaluate implausibility over parameter volume
                for i in range(len(theta_test)):
                    
                    implausibility_all[i, x+(Nx*m)] = implausibility(mu[i], obs_data[m, x], sd[i], var_method, var_exp)

        # choose maximum implausibility
        # create function to do this separately later, and look at 2nd or 3rd max
        # get index of maximum implaus for all outputs
        #max_I = np.argmax(implausibility_all, axis=1)
        # get index of second highest maximum implaus for all outputs
        max2_I = implausibility_all.argsort()[:,-2]
        implausibilities = implausibility_all[range(len(max2_I)), max2_I]

        # identify nonimplausible region
        samples_implaus = np.concatenate((theta_test, implausibilities.reshape(-1,1)), axis=1)
        nonimplausible = np.delete(samples_implaus, np.where(samples_implaus[:,-1] > 3), axis=0)

        ctr, cov, ell = find_bounding_ellipse(nonimplausible[:,:ndim], 0)
        
        # plot implausibilities and optical depth
        variable_names = ['theta1', 'theta2', 'theta3']
        for i in range(ndim):
            for j in range(ndim):
                ax = p1axes[i,j]
                variables = [i,j]
                if i == j:
                    #plot.optical_depth_1D(samples_implaus, 20, ax, fig, i, variable_names[i], parameter_bounds_initial)
                    #ax_right = ax.twinx()
                    #theta_vals = np.linspace(parameter_bounds_initial[i,0], parameter_bounds_initial[i,1], 100)
                    #for m in range(len(true_model)):
                        #ax_right.plot(theta_vals, stats.norm.pdf(theta_vals, true_thetas[m][i], np.sqrt(H[m][i,i])), color=color_list[m])
                    if i < ndim-1:
                        ax.scatter(nonimplausible[:,i], nonimplausible[:,i+1])
                        ax.set_xlim([parameter_bounds_initial[i,0], parameter_bounds_initial[i,1]])
                        ax.set_ylim([parameter_bounds_initial[i+1,0], parameter_bounds_initial[i+1,1]])
                        covi = np.array([[cov[i,i], cov[i,i+1]],[cov[i+1,i], cov[i+1,i+1]]])
                        plot.get_cov_ellipse(covi, [ctr[i],ctr[i+1]], 1, ax, color='red')
                    else:
                        ax.scatter(nonimplausible[:,0], nonimplausible[:,i], s=2)
                        ax.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
                        ax.set_ylim([parameter_bounds_initial[i,0], parameter_bounds_initial[i,1]])
                        covi = np.array([[cov[i,i], cov[i,0]],[cov[0,i], cov[0,0]]])
                        plot.get_cov_ellipse(covi, [ctr[i],ctr[0]], 1, ax, color='red')
                    

                elif i > j:
                    #plot.implausibility_2D(samples_implaus, parameter_bounds_initial, ax1, fig1, k, n, variables, [variable_names[j], variable_names[i]])
                    plot.implausibility(samples_implaus, parameter_bounds_initial, ax , fig, k, 0, [j,i], 
                            [variable_names[j], variable_names[i]], bins=30)
                    for m in range(len(true_model)):
                        ax.scatter(true_thetas[m][j],true_thetas[m][i], color='red', marker='x')
                #ax1.scatter(true_parameters[j], true_parameters[i], color='red', marker='x', label='Observed Data' if n == 0 else "")
                else:
                    None
                    #cov_matrix2 = np.array([[cov[i,i], cov[i,j]],[cov[j,i], cov[j,j]]])
                    #plot.get_cov_ellipse(cov_matrix2, [ctr[i],ctr[j]], 1, ax, 'red')
                    #for m in range(len(true_model)):
                        #cov_matrix = np.array([[H[m][i,i], H[m][i,j]],[H[m][j,i], H[m][j,j]]])
                        #plot.get_cov_ellipse(cov_matrix, [true_thetas[m][i],true_thetas[m][j]], 3, ax, color_list[m])
                        #ax.set_xlim([parameter_bounds_initial[i,0], parameter_bounds_initial[i,1]])
                        #ax.set_ylim([parameter_bounds_initial[j,0], parameter_bounds_initial[j,1]])
                        #plot.optical_depth_2D(samples_implaus, parameter_bounds_initial, ax1, fig1, k, n, variables, [variable_names[i], variable_names[j]])
        
        
        if nonimplausible.size == 0:
                print('empty')
                continue

        # identify nonimplausible region boundaries and plot
        parameter_bounds = np.zeros((ndim, 2))
        for i in range(ndim):
            parameter_bounds[i,0] = nonimplausible[:,i].min()
            parameter_bounds[i,1] = nonimplausible[:,i].max()

    return ctrlist, covlist, nonimplausible, ell

            
