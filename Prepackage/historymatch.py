import numpy as np
import emulator
import matplotlib.pyplot as plt

# define implausibility measure
def implausibility(E, z, var, var_md, var_obs):
    
    # E - emulator expectation
    # z - observational data
    # var - credible interval
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


def plot(x,y,ax):
    return ax.scatter(x,y)
