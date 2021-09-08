import numpy as np
import emulator
import matplotlib.pyplot as plt
import random

import data

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


def evaluate_cell(cell, n_grid, parameter_bounds, points, ndim):

    cell_volume = 1

    previous_indices = np.array([])

    for i in range(2):
        i_range = parameter_bounds[i,1] - parameter_bounds[i,0]
        cell_i = i_range / n_grid
        cell_volume = cell_volume * cell_i

        min_i = parameter_bounds[i,0] + cell_i*(cell[i]%n_grid)
        max_i = parameter_bounds[i,0] + cell_i*(cell[i]%n_grid + 1)

        indices = np.where(np.logical_and(points[:,i]>=min_i, points[:,i]<=max_i))

        #all_indices = np.append(all_indices, indices_temp[0])
        common_indices = np.intersect1d(indices, previous_indices)
        previous_indices = indices

    #print(common_indices)
    density = len(common_indices) / cell_volume
    return common_indices, density

def find_clusters_2D(dataset, input_test, ndim, n_grid = 5, threshold=1):

    # define parameter space
    parameter_bounds = data.locate_boundaries(input_test, ndim)

    cluster = np.empty((0,2))
    cluster_list = []
    points = np.delete(dataset, -1, axis=1)
    # divide data into grid
    N_cells = n_grid**ndim
    traversed_cells = []
 
    while len(traversed_cells) <= N_cells:
        # randomly select cell
        cell = random.randint(0,N_cells)
        # check if in list. if so, generate another
        while cell in traversed_cells:
            cell = random.randint(0,N_cells)
        # calculate density of cell
        traversed_cells.append(cell)
        cell_pts, density =  evaluate_cell(cell, n_grid, parameter_bounds, points, ndim)
        # check if density is greater than threshold
        if density > threshold:
            # mark cell as new cluster
            cluster = np.append(cluster, points[cell_pts], axis=0)
            # calculate density of neighbouring cells
            neighbours = [cell+1, cell-1, cell+n_grid, cell-n_grid]
            while len(neighbours) != 0:
                for neighbour in neighbours:
                    # if cell not checked already, add to traversed list
                    if neighbour not in traversed_cells:
                        traversed_cells.append(neighbour)
                        # check density of neighbouring cell
                        cell_pts_neigh, density_neigh = evaluate_cell(neighbour, n_grid, parameter_bounds, points, ndim)
                        if density_neigh > threshold:
                            cluster = np.append(cluster, points[cell_pts_neigh], axis=0)
                            neighbours.extend([neighbour+1, neighbour-1, neighbour+n_grid, neighbour-n_grid])
                            neighbours.remove(neighbour)
                            #print('neighbour: ' + str(neighbour))
                        else:
                            neighbours.remove(neighbour)
                    else:
                        neighbours.remove(neighbour)
            else:
                cluster_list.append(cluster)
                cluster = np.empty((0,2))
                
        else:
            pass
    
    return cluster_list

def find_clusters_3D(dataset, input_test, ndim, parameter_bounds, n_grid = 10, threshold=2):

    # define parameter space
    parameter_bounds = data.locate_boundaries(input_test, ndim)

    cluster = np.empty((0, ndim))
    cluster_list = []
    points = np.delete(dataset, -1, axis=1)

    cellsides = [np.arange(0,n_grid,1) for dim in range(ndim)]
    all_cells = np.array(np.meshgrid(*cellsides)).T.reshape(-1,ndim)
    # divide data into grid
    N_cells = n_grid**ndim
    traversed_cells = []
 
    while len(traversed_cells) < N_cells:

        # randomly select cell
        cell_index = np.random.randint(0, N_cells)
        cell = all_cells[cell_index]
        # check if in list. if so, generate another
        while cell_index in traversed_cells:
            cell_index = np.random.randint(0, N_cells)
            cell = all_cells[cell_index]
        # calculate density of cell
        traversed_cells.append(cell_index)
        cell_pts, density =  evaluate_cell(cell, n_grid, parameter_bounds, points, ndim)
        # check if density is greater than threshold
        if density > threshold:
            # mark cell as new cluster
            cluster = np.append(cluster, points[cell_pts], axis=0)
            # calculate density of neighbouring cells
            neighbours = []
            for j in range(ndim):
                n_1 = [val for val in cell]
                n_2 = [val for val in cell]
                n_1[j] += 1
                n_2[j] += -1
                n1_index = np.where((all_cells == n_1).all(axis=1))[0]
                n2_index = np.where((all_cells == n_1).all(axis=1))[0]
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
                        cell_pts_neigh, density_neigh = evaluate_cell(neighbour, n_grid, parameter_bounds, points, ndim)
                        if density_neigh > threshold:
                            cluster = np.append(cluster, points[cell_pts_neigh], axis=0)
                            for j in range(ndim):
                                n_1 = [val for val in cell]
                                n_2 = [val for val in cell]
                                n_1[j] += 1
                                n_2[j] += -1
                                n1_index = np.where((all_cells == n_1).all(axis=1))[0]
                                n2_index = np.where((all_cells == n_1).all(axis=1))[0]
                                if (n1_index not in traversed_cells) & (n1_index.size != 0):
                                    neighbours.append(n_1)
                                if (n2_index not in traversed_cells) & (n2_index.size != 0):
                                    neighbours.append(n_2)

                            neighbours.remove(neighbour)
                            #print('neighbour: ' + str(neighbour))
                        else:
                            neighbours.remove(neighbour)
                    else:
                        neighbours.remove(neighbour)
            else:
                cluster_list.append(cluster)
                cluster = np.empty((0,ndim))
                
        else:
            pass
    return cluster_list