import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from scipy import stats
from scipy.stats import norm, uniform
import scipy.optimize


def plot_implausibility2D(samples, parameter_bounds, parameters, bins=20, Fig=None, colorbar=False, 
                            labels=None, plot_kwargs=None):

    """
    Plots a minimised implausibilty plot for two parameters.

    Args
    ----

    samples : array of floats
        (nsamples x ndim+1) size array of nsamples coordinates in parameter space and
        each coordinate's corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.

    parameter_bounds : array of floats
        (ndim x 2) array containing the initial upper and lower bounds assigned to 
        each parameter to be plotted.

    parameters : array of ints
        If ndim > 2, a (1 x 2) array of the parameters to plot is required. The two
        values in the array are integers corresponding to parameters, following the
        same order as parameter_bounds. The first variable in the array will be plotted
        on the x-axis and the second variable on the y-axis.
    
    bins : int, optional
        Number of bins in which to divide parameter space by. Increases resolution of
        plot. Default is 20.

    Fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, plots on to the provided axis and figure. Default is None and will
        instead initialise a figure.

    labels : (iterable (ndim,))
        List of parameter names.

    plot_kwargs : dict, optional
        Additional keyword arguments passed to 'axes.contourf'.

    """


    if Fig == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = Fig

    # split axis into bins
    ybound0 = parameter_bounds[1, 0]
    ybound1 = parameter_bounds[1, 1]
    xbound0 = parameter_bounds[0, 0]
    xbound1 = parameter_bounds[0, 1]
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
            y_indices = np.where(np.logical_and(samples[:,parameters[1]]>=ybin0, samples[:,parameters[1]]<ybin1))
            x_indices = np.where(np.logical_and(samples[:,parameters[0]]>=xbin0, samples[:,parameters[0]]<xbin1))
            bin_indices = np.intersect1d(x_indices, y_indices)
            # find minimised implausibility over remaining dimensions
            if bin_indices.shape[0] != 0:
                implausibilities[i,j] = samples[bin_indices][:,-1].min()
            else:
                implausibilities[i,j] = np.NaN

    
    clevels=[0,0.5,1,1.5,2,2.5,3,3.5]

    if plot_kwargs is None:
        plot_kwargs = dict()
    if plot_kwargs.get('levels') is None:
        plot_kwargs['levels'] = plot_kwargs.get('levels', clevels)

    if plot_kwargs.get('cmap') is None:
        colormap = 'viridis_r'
        cmap = cm.get_cmap(colormap, len(clevels))
        colorlist = [cmap(0), cmap(0.1), cmap(0.15), cmap(0.3), cmap(0.45), cmap(0.6), cmap(0.85), cmap(1)]
        plot_kwargs['colors'] = plot_kwargs.get('colors', colorlist)
    else:
        cmap = cm.get_cmap(plot_kwargs.get('cmap'))

    ax.set_facecolor(cmap(0.9999))


    im = ax.contourf(xvals, yvals, implausibilities, **plot_kwargs)

    
    ax.set_xlim([parameter_bounds[0,0], parameter_bounds[0,1]])
    ax.set_ylim([parameter_bounds[1,0], parameter_bounds[1,1]])
    
    if labels:
        ax.set_xlabel(labels[parameters[0]])
        ax.set_ylabel(labels[parameters[1]])
    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label('Implausibility')
    #if colorbar == False:
        #cbar.remove()
    #ax.set_title('Wave ' + str(wave+1) + ' Implausibility')

    


def opticaldepth_2D(samples, parameter_bounds, parameters, bins=20, Fig=None, colorbar=False, labels=None, plot_kwargs=None):
    
    """
    Plots an optical depth plot for two parameters.

    Args
    ----

    samples : array of floats
        (nsamples x ndim+1) size array of nsamples coordinates in parameter space and
        each coordinates corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.

    parameter_bounds : array of floats
        (ndim x 2) array containing the initial upper and lower bounds assigned to 
        each parameter to be plotted.

    parameters : array of ints
        If ndim > 2, a (1 x 2) array of the parameters to plot is required. The two
        values in the array are integers corresponding to parameters, following the
        same order as parameter_bounds. The first variable in the array will be plotted
        on the x-axis and the second variable on the y-axis.
    
    bins : int, optional
        Number of bins in which to divide parameter space by. Increases resolution of
        plot. Default is 20.

    Fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, plots on to the provided axis and figure. Default is None and will
        instead initialise a figure.

    labels : (iterable (ndim,))
        List of parameter names.

    plot_kwargs : dict, optional
        Additional keyword arguments passed to 'axes.contourf'.

    """


    if Fig == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = Fig

    # split axis into bins
    ybound0 = parameter_bounds[1, 0]
    ybound1 = parameter_bounds[1, 1]
    xbound0 = parameter_bounds[0, 0]
    xbound1 = parameter_bounds[0, 1]
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
            y_indices = np.where(np.logical_and(samples[:,parameters[1]]>=ybin0, samples[:,parameters[1]]<ybin1))
            x_indices = np.where(np.logical_and(samples[:,parameters[0]]>=xbin0, samples[:,parameters[0]]<xbin1))
            bin_indices = np.intersect1d(x_indices, y_indices)
            # count nonimplausible points over remaining dimensions
            n_pts = np.count_nonzero(samples[bin_indices][:,-1] < 3)
            # find initial number of points 
            y_indices_0 = np.where(np.logical_and(samples[:,parameters[1]]>=ybin0, samples[:,parameters[1]]<ybin1))
            x_indices_0 = np.where(np.logical_and(samples[:,parameters[0]]>=xbin0, samples[:,parameters[0]]<xbin1))
            bin_indices_0 = np.intersect1d(x_indices_0, y_indices_0)

            if len(bin_indices_0) != 0:
                densities[i,j] = n_pts/len(bin_indices_0)
            else:
                densities[i,j] = 0
            

    
    #bounds=np.linspace(np.amin(densities), np.amax(densities), 8)
    clevels = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6]

    if plot_kwargs is None:
        plot_kwargs = dict()
    
    plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'magma')

    cmap = cm.get_cmap(plot_kwargs.get('cmap'))

    im = ax.contourf(xvals, yvals, densities, **plot_kwargs)

    ax.set_facecolor(cmap(0))


    ax.set_xlim([parameter_bounds[0,0], parameter_bounds[0,1]])
    ax.set_ylim([parameter_bounds[1,0], parameter_bounds[1,1]])
    if labels:
        ax.set_xlabel(labels[parameters[0]])
        ax.set_ylabel(labels[parameters[1]])




def plotcorner(samples, parameter_bounds, ndim, bins=20, Fig=None, colorbar=False,
                show_axes = 'all', labels=None, plot1D_kwargs=None, plot2D_kwargs=None):

    """
    Plots a corner plot of 2-d implausibility plots for all parameters,
    for a particular wave.

    Args
    ----
    samples : array of floats
        (nsamples x ndim+1) size array of nsamples coordinates in parameter space and
        each coordinates corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.jh

    parameter_bounds : array of floats
        (ndim x 2) array containing the initial upper and lower bounds assigned to 
        each parameter to be plotted.

    bins : int, optional
        Number of bins in which to divide parameter space by. Increases resolution of
        plot. Default is 20.

    Fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, plots on to the provided axis and figure. Default is None and will
        instead initialise a figure.

    colorbar : bool
        Option to include a colorbar in the plot. Default is False.

    show_axes : str
        'all' shows all axes and labels. 'edge' shows axes and labels of outer edges.
        'None' shows no axes and labels.

    labels : (iterable (ndim,))
        List of parameter names.

    plot1D_kwargs : dict, optional
        Additional keyword arguments passed to 'axes.bar' for use in the 1d optical
        depth plots.

    plot2D_kwargs : dict, optional
        Additional keyword arguments passed to 'axes.contourf' for use in the 2d plots.
        

    
    """

    if Fig == None:
        fig, axes = plt.subplots(ndim,ndim,figsize=(10,10))
    else:
        fig, axes = Fig

    for j in range(ndim):
        for i in range(ndim):
            ax = axes[j,i]
            #fig.add_subplot(ndim, ndim, (1+i+ndim*j), sharey='row')
            if show_axes == None:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            parameters = [i,j]
            if i == j:
                ax.axes.yaxis.set_visible(False)
                if labels != None:
                    label=labels[i]
                else:
                    label=None
                opticaldepth_1D(samples, parameter_bounds[i], parameter=i, bins=bins, Fig=(fig,ax), label=label)
            elif i < j:
                plot_implausibility2D(samples, parameter_bounds[[i,j]], [i,j], bins, (fig,ax), colorbar=colorbar, labels=labels, plot_kwargs=plot1D_kwargs)
            else:
                opticaldepth_2D(samples, parameter_bounds[[i,j]], [i,j], bins, (fig,ax), colorbar=colorbar, labels=labels, plot_kwargs=plot2D_kwargs)
    
    plt.tight_layout()




def opticaldepth_1D(samples, parameter_bounds, parameter, bins=20, Fig=None, label=None, plot_kwargs=None):

    """
    Plots the optical depth for a single parameter, for a specified wave.

    Args
    ----

    samples : array of floats
        (nsamples x ndim+1) size array of nsamples coordinates in parameter space and
        each coordinates corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.

    parameter_bounds : array of floats
        (1 x 2) array containing the initial upper and lower bounds assigned to 
        the parameter to be plotted.

    parameter : int
        The index of the parameter to be plotted, corresponding to a column in samples.
    
    bins : int, optional
        Number of bins in which to divide parameter space by, in order to evaluate density
        of points. Default is 20.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, plots on to the provided axis and figure. Default is None and will
        instead initialise a figure.

    label : str, optional
        Name of parameter

    plot_kwargs : dict, optional
        Additional keyword arguments passed to 'axes.bar'.
    
    """

    if Fig == None:
        fig = plt.figure()
    else:
        fig, ax = Fig

    

    # split axis into bins
    xmin = parameter_bounds[0]
    xmax = parameter_bounds[1]
    bin_size = (xmax - xmin)/bins

    counts = []
    initial_density = 0

    for i in range(bins):
        # define bin
        bin_min = xmin + i*bin_size
        bin_max = bin_min + bin_size
        # find values within bin
        indices = np.where(np.logical_and(samples[:,parameter]>=bin_min, samples[:,parameter]<bin_max))
        # find initial density in this bin
        density_0 = len(indices)
        # for each bin, count nonimplausible values over remaining parameter volume
        bin_count = np.count_nonzero(samples[indices,-1] < 3, axis=1)
        if density_0 != 0:
            counts.append(bin_count[0]/density_0)
        else:
            counts.append(np.NanN)

    if plot_kwargs is None:
        plot_kwargs = dict()

    plot_kwargs['width'] = plot_kwargs.get('width', bin_size)
    plot_kwargs['align'] = plot_kwargs.get('align', 'edge')

    ax.bar(np.arange(xmin, xmax, bin_size), counts, **plot_kwargs)

    if label:
        ax.set_xlabel(label)
