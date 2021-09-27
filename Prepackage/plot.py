import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from scipy import stats
from scipy.stats import norm, uniform
import scipy.optimize

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
                implausibilities[i,j] = input_imp[bin_indices][:,-1].min()
            else:
                implausibilities[i,j] = np.NaN

    #if np.nanmax(implausibilities) > 3.5:
        #bounds=[0,0.5,1,1.5,2,2.5,3,3.5]
    #else:
        #bounds = np.linspace(np.nanmin(implausibilities), np.nanmax(implausibilities), 8)
    bounds=[0,0.5,1,1.5,2,2.5,3,3.5]
    cmap = cm.get_cmap('viridis_r', len(bounds))

    ax.contourf(xvals, yvals, implausibilities, levels=bounds, colors=[cmap(0), cmap(0.1), cmap(0.15), cmap(0.3), cmap(0.45), cmap(0.6), cmap(0.85), cmap(1)])
    #im = ax.contourf(xvals, yvals, implausibilities, cmap='viridis_r')


    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([parameter_bounds_initial[0,0], parameter_bounds_initial[0,1]])
    ax.set_ylim([parameter_bounds_initial[1,0], parameter_bounds_initial[1,1]])
    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label('Implausibility')
    #im.set_clim(0,10)
    ax.set_ylabel(variable_names[1])
    ax.set_xlabel(variable_names[0])
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



def optical_depth_1D(input_imp, bins, ax, fig, variable, variable_name, parameter_bounds_initial):
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

    ax.bar(bin_list, counts, alpha=0.5)


    if variable != input_imp.shape[1] - 2:
        ax.axes.xaxis.set_visible(False)
    if variable != 0:
        ax.axes.yaxis.set_visible(False)
    ax.set_xlim([-2,2])

def optical_depth_2D(input_imp, parameter_bounds_initial, ax, fig, wave, variables, variable_names, bins=30):
    # split axis into bins
    ybound0 = parameter_bounds_initial[variables[1], 0]
    ybound1 = parameter_bounds_initial[variables[1], 1]
    xbound0 = parameter_bounds_initial[variables[0], 0]
    xbound1 = parameter_bounds_initial[variables[0], 1]
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
            y_indices = np.where(np.logical_and(input_imp[:,variables[1]]>=ybin0, input_imp[:,variables[1]]<ybin1))
            x_indices = np.where(np.logical_and(input_imp[:,variables[0]]>=xbin0, input_imp[:,variables[0]]<xbin1))
            bin_indices = np.intersect1d(x_indices, y_indices)
            # count nonimplausible points over remaining dimensions
            n_pts = np.count_nonzero(input_imp[bin_indices][:,-1] < 3)
            # find initial number of points 
            y_indices_0 = np.where(np.logical_and(input_imp[:,variables[1]]>=ybin0, input_imp[:,variables[1]]<ybin1))
            x_indices_0 = np.where(np.logical_and(input_imp[:,variables[0]]>=xbin0, input_imp[:,variables[0]]<xbin1))
            bin_indices_0 = np.intersect1d(x_indices_0, y_indices_0)

            if len(bin_indices_0) != 0:
                densities[i,j] = n_pts/len(bin_indices_0)
            else:
                densities[i,j] = 0
            

    
    #bounds=np.linspace(np.amin(densities), np.amax(densities), 8)
    bounds = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6]
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
    #ax.set_ylabel(variable_names[1])
    #ax.set_xlabel(variable_names[0])
    #cbar.remove()
    #ax.set_title('Wave ' + str(wave+1) + ' Implausibility')
    #if variables[0] != input_imp.shape[1] - 2:
        #ax.axes.xaxis.set_visible(False)
    #if variables[1] != 0:
        #ax.axes.yaxis.set_visible(False)
    ax.axis('off')

def plot_ellipsoid(ell, ax):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))
    print(np.linalg.inv(ell.a))
    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i,j], y[i,j] = ell.ctr + np.dot(ell.axes,
                                                      [x[i,j],y[i,j]])

    ax.plot(x, y, color='#2980b9', alpha=0.2)

def get_cov_ellipse(cov, centre, nstd, ax, color):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(np.abs(eigvals))
    
    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([0.5*width*np.cos(t) , 0.5*height*np.sin(t)]) 
    R_rot = np.array([[np.cos(theta) , -np.sin(theta)],[np.sin(theta), np.cos(theta)]])  
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    
    ax.plot(centre[0]+Ell_rot[0,:] , centre[1]+Ell_rot[1,:], color=color)
    ax.scatter(centre[0], centre[1], marker='x', color=color)
        
    #return Ellipse(xy=centre, width=width, height=height,
                   #angle=np.degrees(theta), **kwargs)


def plot_ellipses(fig, parameter_bounds, true_parameters, H, theta_best, theta_vals, color):
    theta_names = [r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$']
    N = len(theta_best)
    for i in range(N):
        for j in range(N):
            ax = fig.axes[i + N*j]
            if i != 0:
                ax.axes.yaxis.set_visible(False)
                
            if i == j:
                ax.plot(theta_vals[i], stats.norm.pdf(theta_vals[i], theta_best[i], np.sqrt(H[i,i])), color=color)
                ax.set_title(str(theta_names[i]) + '=' + str(round(theta_best[i], 2)), fontsize=14)
                
            elif i < j:
                
                cov_matrix = np.array([[H[i,i], H[i,j]],[H[j,i], H[j,j]]])
                get_cov_ellipse(cov_matrix, [theta_best[i], theta_best[j]], 3, ax, color)
                ax.set_ylabel(theta_names[j])
                ax.set_xlabel(theta_names[i])
                ax.set_xlim([parameter_bounds[i,0], parameter_bounds[i,1]])
                ax.set_ylim([parameter_bounds[j,0], parameter_bounds[j,1]])
                
            else:
                ax.axis('off')