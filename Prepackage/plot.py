import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

def implausibility_2D(input_imp, parameter_bounds_initial, ax, fig, wave, region, variables, variable_names, bins=20):
    
    # split axis into bins
    ybound0 = parameter_bounds_initial[variables[1], 0]
    ybound1 = parameter_bounds_initial[variables[1], 1]
    xbound0 = parameter_bounds_initial[variables[0], 0]
    xbound1 = parameter_bounds_initial[variables[0], 1]
    bin_width = (xbound1 - xbound0)/bins
    bin_height = (ybound1 - ybound0)/bins

    implausibilities = np.zeros((bins, bins))
    xvals = np.linspace(xbound0,xbound1,bins)
    yvals = np.linspace(ybound0,ybound1,bins)

    for i in range(bins): # across y
        ybin0 = ybound0 + i*bin_height
        ybin1 = ybin0 + bin_height
        # add mid of cell to list for plotting
        for j in range(bins):  # across x
            xbin0 = xbound0 + j*bin_width
            xbin1 = xbin0 + bin_height

            # find points within bin
            y_indices = np.where(np.logical_and(input_imp[:,variables[1]]>=ybin0, input_imp[:,variables[1]]<ybin1))
            x_indices = np.where(np.logical_and(input_imp[:,variables[0]]>=xbin0, input_imp[:,variables[0]]<xbin1))
            bin_indices = np.intersect1d(x_indices, y_indices)
            # find minimised implausibility over remaining dimensions
            if bin_indices.shape[0] != 0:
                implausibilities[i,j] = input_imp[bin_indices][:,-1].mean()
            else:
                implausibilities[i,j] = np.NaN

    if np.nanmax(implausibilities) > 3.5:
        bounds=[0,0.5,1,1.5,2,2.5,3,3.5]
    else:
        bounds = np.linspace(np.nanmin(implausibilities), np.nanmax(implausibilities), 8)
    cmap = cm.get_cmap('viridis_r', len(bounds))

    im = ax.contourf(xvals, yvals, implausibilities, levels=bounds, colors=[cmap(0), cmap(0.1), cmap(0.15), cmap(0.3), cmap(0.45), cmap(0.6), cmap(0.85), cmap(1)])
    #im = ax.contourf(xvals, yvals, implausibilities, cmap='viridis_r')


    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
    ax.set_ylim([parameter_bounds_initial[1,0], parameter_bounds_initial[1,1]])
    if region == 0:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Implausibility')
        im.set_clim(0,10)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_title('Wave ' + str(wave+1) + ' Implausibility')

def implausibility(input_imp, parameter_bounds_initial, ax, fig, wave, region, variables, variable_names, bins=20):

    # split axis into bins
    ybound0 = parameter_bounds_initial[variables[0], 0]
    ybound1 = parameter_bounds_initial[variables[0], 1]
    xbound0 = parameter_bounds_initial[variables[1], 0]
    xbound1 = parameter_bounds_initial[variables[1], 1]
    bin_width = (xbound1 - xbound0)/bins
    bin_height = (ybound1 - ybound0)/bins

    implausibilities = np.zeros((bins, bins))
    xvals = np.linspace(xbound0,xbound1,bins)
    yvals = np.linspace(ybound0,ybound1,bins)

    for i in range(bins): # across y
        ybin0 = ybound0 + i*bin_height
        ybin1 = ybin0 + bin_height
        # add mid of cell to list for plotting
        for j in range(bins):  # across x
            xbin0 = xbound0 + j*bin_width
            xbin1 = xbin0 + bin_height

            # find points within bin
            y_indices = np.where(np.logical_and(input_imp[:,variables[0]]>=ybin0, input_imp[:,variables[0]]<ybin1))
            x_indices = np.where(np.logical_and(input_imp[:,variables[1]]>=xbin0, input_imp[:,variables[1]]<xbin1))
            bin_indices = np.intersect1d(x_indices, y_indices)
            # find minimised implausibility over remaining dimensions
            if bin_indices.shape[0] != 0:
                implausibilities[i,j] = input_imp[bin_indices][:,-1].min()
            else:
                implausibilities[i,j] = np.NaN

    if np.nanmax(implausibilities) > 3.5:
        bounds=[0,0.5,1,1.5,2,2.5,3,3.5]
    else:
        bounds = np.linspace(np.nanmin(implausibilities), np.nanmax(implausibilities), 8)
    cmap = cm.get_cmap('viridis_r', len(bounds))

    im = ax.contourf(xvals, yvals, implausibilities, levels=bounds, colors=[cmap(0), cmap(0.1), cmap(0.15), cmap(0.3), cmap(0.45), cmap(0.6), cmap(0.85), cmap(1)])
    #im = ax.contourf(xvals, yvals, implausibilities, cmap='viridis_r')


    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
    ax.set_ylim([parameter_bounds_initial[1,0], parameter_bounds_initial[1,1]])
    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label('Implausibility')
    #im.set_clim(0,10)
    ax.set_ylabel(variable_names[0])
    ax.set_xlabel(variable_names[1])
    #cbar.remove()
    #ax.set_title('Wave ' + str(wave+1) + ' Implausibility')
    #if input_imp.shape[1] > 2:
        #if variables[0] != input_imp.shape[1] - 2:
            #ax.axes.xaxis.set_visible(False)
        #if variables[1] != 0:
            #ax.axes.yaxis.set_visible(False)
    
    

def nonimplausible_pts(input_train, nonimplausible, ax):
# plot nonimplausible datapoints

    # find nonimplausible boundaries
    xmin = nonimplausible[:,0].min()
    xmax = nonimplausible[:,0].max()
    ymin = nonimplausible[:,1].min()
    ymax = nonimplausible[:,1].max()

    ax.scatter(nonimplausible[:,0], nonimplausible[:,1], s=2)
    ax.set_title('Remaining Non-Implausible Datapoints')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.scatter(input_train[:,0], input_train[:,1], marker='x', color='black', label='Training Data')
    ax.axhline(ymin, linestyle = '--', color='pink')
    ax.axhline(ymax, linestyle = '--', color='pink')
    ax.axvline(xmin, linestyle = '--', color='pink')
    ax.axvline(xmax, linestyle = '--', color='pink')
    ax.legend(loc='lower left')



def optical_depth_1D(input_imp, bins, ax, fig, variable, variable_name, parameter_bounds_initial, input_test_0):
    #print(nonimplausible[:,variable])
    #ax.hist(nonimplausible[:,variable], bins=ngrid)

    # split axis into bins
    bound0 = parameter_bounds_initial[variable, 0]
    bound1 = parameter_bounds_initial[variable, 1]
    bin_size = (bound1 - bound0)/bins


    bin_list = []
    counts = []
    initial_density = 0

    for i in range(bins):
        # define bin
        bin0 = bound0 + i*bin_size
        bin1 = bin0 + bin_size
        bin_list.append(bin0 + (bin1-bin0)/2)
        # find values within bin
        indices = np.where(np.logical_and(input_imp[:,variable]>=bin0, input_imp[:,variable]<bin1))
        # find initial density in this bin
        initial_space = len(indices)
        # for each bin, count nonimplausible values over remaining parameter volume
        bin_count = np.count_nonzero(input_imp[indices,-1] < 3, axis=1)
        counts.append(bin_count[0]/initial_space)

    ax.bar(bin_list, counts)


    if variable != input_imp.shape[1] - 2:
        ax.axes.xaxis.set_visible(False)
    if variable != 0:
        ax.axes.yaxis.set_visible(False)
    ax.set_xlim([-2,2])

def optical_depth_2D(input_imp, parameter_bounds_initial, input_test_0, ax, fig, wave, region, variables, variable_names, bins=20):
    # split axis into bins
    ybound0 = parameter_bounds_initial[variables[0], 0]
    ybound1 = parameter_bounds_initial[variables[0], 1]
    xbound0 = parameter_bounds_initial[variables[1], 0]
    xbound1 = parameter_bounds_initial[variables[1], 1]
    bin_width = (xbound1 - xbound0)/bins
    bin_height = (ybound1 - ybound0)/bins

    densities = np.zeros((bins, bins))
    xvals = np.linspace(xbound0,xbound1,bins)
    yvals = np.linspace(ybound0,ybound1,bins)

    for i in range(bins): # across y
        ybin0 = ybound0 + i*bin_height
        ybin1 = ybin0 + bin_height
        # add mid of cell to list for plotting
        for j in range(bins):  # across x
            xbin0 = xbound0 + j*bin_width
            xbin1 = xbin0 + bin_height

            # find points within bin
            y_indices = np.where(np.logical_and(input_imp[:,variables[0]]>=ybin0, input_imp[:,variables[0]]<ybin1))
            x_indices = np.where(np.logical_and(input_imp[:,variables[1]]>=xbin0, input_imp[:,variables[1]]<xbin1))
            bin_indices = np.intersect1d(x_indices, y_indices)
            # count nonimplausible points over remaining dimensions
            n_pts = np.count_nonzero(input_imp[bin_indices][:,-1] < 3)
            # find initial number of points 
            y_indices_0 = np.where(np.logical_and(input_imp[:,variables[0]]>=ybin0, input_imp[:,variables[0]]<ybin1))
            x_indices_0 = np.where(np.logical_and(input_imp[:,variables[1]]>=xbin0, input_imp[:,variables[1]]<xbin1))
            bin_indices_0 = np.intersect1d(x_indices_0, y_indices_0)

            if len(bin_indices_0) != 0:
                densities[i,j] = n_pts/len(bin_indices_0)
            else:
                densities[i,j] = 0
            

    
    #bounds=np.linspace(np.amin(densities), np.amax(densities), 8)
    bounds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.8]
    cmap = cm.get_cmap('magma', len(bounds))
    #print(np.amin(densities))
    #print(np.amax(densities))
    im = ax.contourf(xvals, yvals, densities, levels=bounds, colors=[cmap(0), cmap(0.1), cmap(0.15), cmap(0.3), cmap(0.45), cmap(0.6), cmap(0.85), cmap(1)])
    #im = ax.contourf(xvals, yvals, densities, cmap='viridis_r')


    #im =  ax.tricontourf(input_test[:,variables[1]],input_test[:,variables[0]],implaus, levels=[0, 0.5, 1, 2.5, 2.6, 2.7, 3], cmap='viridis_r')

    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
    ax.set_ylim([parameter_bounds_initial[1,0], parameter_bounds_initial[1,1]])
    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label('Implausibility')
    #im.set_clim(0,10)
    ax.set_ylabel(variable_names[0])
    ax.set_xlabel(variable_names[1])
    #cbar.remove()
    #ax.set_title('Wave ' + str(wave+1) + ' Implausibility')
    if variables[0] != input_imp.shape[1] - 2:
        ax.axes.xaxis.set_visible(False)
    if variables[1] != 0:
        ax.axes.yaxis.set_visible(False)