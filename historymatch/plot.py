import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.stats import norm

def plot_implausibility2D(samples, parameter_bounds, parameters, bins=20, Fig=None, colorbar=False, 
                            labels=None, plot_kwargs=None):

    """
    Plots a minimised implausibilty plot for two parameters.

    Args
    ----

    samples : ndarray, shape (N, ndim+1)
        N sample coordinates in parameter space and
        each samples's corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.

    parameter_bounds : ndarray, shaPE (ndim, 2)
        array containing the initial upper and lower bounds assigned to 
        each parameter to be plotted.

    parameters : ndarray of ints, shape (1, 2)
        If ndim > 2, an array of the parameters to plot is required. The two
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
            y_indices = np.where(np.logical_and(samples[:,parameters[1]]>=ybin0, samples[:,parameters[1]]<ybin1))[0]
            x_indices = np.where(np.logical_and(samples[:,parameters[0]]>=xbin0, samples[:,parameters[0]]<xbin1))[0]
            bin_indices = np.intersect1d(x_indices, y_indices)
            # find minimised implausibility over remaining dimensions
            if bin_indices.shape[0] != 0:
                implausibilities[i,j] = samples[bin_indices][:,-1].min()
            else:
                implausibilities[i,j] = np.NaN
    
    clevels=[1.6,1.8,2.0,2.2,2.4,2.6,2.8,3,3.2]

    if plot_kwargs is None:
        plot_kwargs = dict()
    if plot_kwargs.get('levels') is None:
        plot_kwargs['levels'] = plot_kwargs.get('levels', clevels)

    if plot_kwargs.get('cmap') is None:
        colormap = 'viridis_r'
        cmap = cm.get_cmap(colormap, len(clevels))
        print(cmap(0.13))
        colorlist = [cmap(0), cmap(0.15), cmap(0.25), cmap(0.35), cmap(0.45), cmap(0.6), cmap(0.75), cmap(0.85), cmap(1), cmap(1)]
        plot_kwargs['colors'] = plot_kwargs.get('colors', colorlist)
    else:
        cmap = cm.get_cmap(plot_kwargs.get('cmap'))

    ax.set_facecolor(cmap(0.9999))


    im = ax.contourf(xvals, yvals, implausibilities, extend="max", **plot_kwargs)

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
    if colorbar ==  True:
        return im

    


def opticaldepth_2D(samples, nonimplausible_samples, parameter_bounds, parameters, bins=20, Fig=None, colorbar=False, labels=None, plot_kwargs=None):
    
    """
    Plots an optical depth plot for two parameters.

    Args
    ----

    samples : ndarray, shape (N, ndim+1)
        N sample coordinates in parameter space and
        each sample's corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.

    parameter_bounds : ndarray, shape (ndim, 2)
        Array containing the initial upper and lower bounds assigned to 
        each parameter to be plotted.

    parameters : array of ints, shape (1, 2)
        If ndim > 2, an array of the parameters to plot is required. The two
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
            y_indices = np.where(np.logical_and(samples[:,parameters[1]]>=ybin0, samples[:,parameters[1]]<ybin1))[0]
            x_indices = np.where(np.logical_and(samples[:,parameters[0]]>=xbin0, samples[:,parameters[0]]<xbin1))[0]
            bin_indices = np.intersect1d(x_indices, y_indices)
            # count nonimplausible points over remaining dimensions
            n_pts = np.count_nonzero(samples[bin_indices][:,-1] < 3)

            # find points within bin
            y_indices = np.where(np.logical_and(nonimplausible_samples[:,parameters[1]]>=ybin0, nonimplausible_samples[:,parameters[1]]<ybin1))[0]
            x_indices = np.where(np.logical_and(nonimplausible_samples[:,parameters[0]]>=xbin0, nonimplausible_samples[:,parameters[0]]<xbin1))[0]
            bin_indices = np.intersect1d(x_indices, y_indices)
            # count nonimplausible points over remaining dimensions
            n_pts = len(bin_indices)

            # find initial number of points 
            y_indices_0 = np.where(np.logical_and(samples[:,parameters[1]]>=ybin0, samples[:,parameters[1]]<ybin1))
            x_indices_0 = np.where(np.logical_and(samples[:,parameters[0]]>=xbin0, samples[:,parameters[0]]<xbin1))
            bin_indices_0 = np.intersect1d(x_indices_0, y_indices_0)

            #nonimplausible_samples

            if len(bin_indices_0) != 0:
                densities[i,j] = n_pts/len(bin_indices_0)
            else:
                densities[i,j] = 0

    #bounds=np.linspace(np.amin(densities), np.amax(densities), 8)
    clevels = [0,0.05,0.1,0.2,0.4,0.6]

    if plot_kwargs is None:
        plot_kwargs = dict()
    
    plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'gist_heat_r')

    levels = plot_kwargs.get('levels')
    cmap = cm.get_cmap(plot_kwargs.get('cmap'), len(clevels))

    im = ax.contourf(xvals, yvals, densities, extend="max", **plot_kwargs)

    #ax.set_facecolor(cmap(0))
    #cbar = fig.colorbar(im)


    ax.set_xlim([parameter_bounds[0,0], parameter_bounds[0,1]])
    ax.set_ylim([parameter_bounds[1,0], parameter_bounds[1,1]])
    if labels:
        ax.set_xlabel(labels[parameters[0]])
        ax.set_ylabel(labels[parameters[1]])

    if colorbar ==  True:
        return im




def plotcorner(samples, parameter_bounds, ndim, bins=20, Fig=None, colorbar=False,
                show_axes = 'all', labels=None, plot1D_kwargs=None, plot2D_kwargs=None):

    """
    Plots a corner plot of 2-d implausibility plots given sample coordinates and a corresponding
    implausibility for each sample.

    Args
    ----
    samples : ndarray, shape (nsamples, ndim+1)
        Array of nsamples coordinates in parameter space and
        each coordinates corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.jh

    parameter_bounds : ndarray, shape (ndim, 2)
        Array containing the initial upper and lower bounds assigned to 
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




def opticaldepth_1D(samples, parameter_bounds, parameter, bins=20, normalize=False, Fig=None, label=None, plot_kwargs=None):

    """
    Plots the optical depth for a single parameter, for a specified wave.

    Args
    ----

    samples : ndarray, shape (nsamples, ndim+1)
        Array of nsamples coordinates in parameter space and
        each coordinates corresponding implausibility. An array of samples for each wave
        is an attribute of the output of HistoryMatch.run(). The samples for each wave
        can be accessed by indexing the array of samples.

    parameter_bounds : ndarray, shape (1,2)
        Array containing the initial upper and lower bounds assigned to 
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
        density_0 = len(indices[0])
        # for each bin, count nonimplausible values over remaining parameter volume
        bin_count = np.count_nonzero(samples[indices,-1] < 3, axis=1)
        if density_0 != 0:
            counts.append(bin_count[0]/density_0)
        else:
            counts.append(0)

    if plot_kwargs is None:
        plot_kwargs = dict()

    #plot_kwargs['width'] = plot_kwargs.get('width', bin_size)
    #plot_kwargs['align'] = plot_kwargs.get('align', 'edge')
    plot_kwargs['where'] = plot_kwargs.get('where', 'mid')
    plot_kwargs['label'] = plot_kwargs.get('label')

    ax.step(np.arange(xmin, xmax, bin_size), counts, **plot_kwargs)

    if label:
        ax.set_xlabel(label)
    if plot_kwargs.get('label'):
        ax.legend(loc='upper right')


def get_cov_ellipse(cov, centre, chisq, ax, color, linestyle, lw=3):
    """
    Plot an isoprobaility-contour ellipse given a mean (centre) and
    2D covariance matrix.

    Args
    ---

    cov : ndarray
        Covariance matrix

    centre : ndarray
        Centre, or mean, of ellipse

    chisq : float
        Critical chisq value (determines probability mass within ellipse)

    ax : obj
        Axis of figure to plot in

    color : str
        Color of ellipse boundary

    linestyle : str
        Linestyle of ellipse boundary (see Matplotlib)

    lw : int
        Linewidth of ellipse boundary (see Matplotlib)


    """

    # Find and sort eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    theta = np.arctan2(eigvecs[:,0][1], eigvecs[:,0][0])

    # Width and height of ellipse
    width, height = 2 * np.sqrt(chisq*np.abs(eigvals))
    
    t = np.linspace(0, 2*np.pi, 100)
    ellipse = np.array([0.5*width*np.cos(t) , 0.5*height*np.sin(t)]) 
    R = np.array([[np.cos(theta) , -np.sin(theta)],[np.sin(theta), np.cos(theta)]])  # rotation matrix
    ellipse_rot = np.zeros((2,ellipse.shape[1]))
    for i in range(ellipse.shape[1]):
        ellipse_rot[:,i] = np.dot(R,ellipse[:,i])

    ax.plot( centre[0]+ellipse_rot[0,:] , centre[1]+ellipse_rot[1,:], linewidth=lw, color=color, linestyle=linestyle)