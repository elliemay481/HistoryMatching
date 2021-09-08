import numpy as np
import matplotlib.pyplot as plt

def implausibility_2D(input_test, implaus, parameter_bounds, ax, fig, wave, region):
    
    im =  ax.tricontourf(input_test[:,0],input_test[:,1],implaus, levels=20, cmap='viridis_r')

    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([parameter_bounds[0,0], parameter_bounds[0,1]])
    ax.set_ylim([parameter_bounds[1,0], parameter_bounds[1,1]])
    if region == 0:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Implausibility')
        im.set_clim(0,10)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_title('Wave ' + str(wave+1) + ' Implausibility')

def implausibility_2D(input_test, implaus, parameter_bounds, ax, fig, wave, region):
    
    im =  ax.tricontourf(input_test[:,0],input_test[:,1],implaus, levels=20, cmap='viridis_r')

    ax.set_facecolor((68/255,1/255,84/255))
    ax.set_xlim([parameter_bounds[0,0], parameter_bounds[0,1]])
    ax.set_ylim([parameter_bounds[1,0], parameter_bounds[1,1]])
    if region == 0:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Implausibility')
        im.set_clim(0,10)
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_title('Wave ' + str(wave+1) + ' Implausibility')
    

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

def optical_depth(nonimplausible, ngrid, ax, fig):
    hist, xedges, yedges = np.histogram2d(nonimplausible[:,1], nonimplausible[:,0], bins=ngrid, range=[[-2, 2], [-2, 2]])

    im = ax.contourf(np.linspace(-2,2,ngrid),np.linspace(-2,2,ngrid),hist/((100/20)**2),cmap='pink')
    cbar = fig.colorbar(im, ax=ax)
    im.set_clim(0,1)