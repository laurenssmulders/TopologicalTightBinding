import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import dirac_string_rotation, energy_difference, gauge_fix_grid
from tight_binding.bandstructure import sort_energy_grid, sort_energy_path
from tight_binding.utilities import rotate

num_points = 100
N = 4
r = 1
c = 1
divergence_threshold_surface = 200
divergence_threshold_boundary = 10
bands = [0,1]
kxmin = 0.1
kxmax = 0.9
kymin = 0.1
kymax = 0.9
L=100


blochvectors = np.identity(3)
blochvectors = blochvectors[np.newaxis,np.newaxis,:,:]
blochvectors = np.repeat(blochvectors, num_points, 0)
blochvectors = np.repeat(blochvectors, num_points, 1)

# 01 TWO NODES AND NOTHING ELSE
if False:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25,0.5]), np.array([0.5,0]), 2, 0.3, num_points)

# 02 ONE DS
if False:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.5,0]), np.array([0,1]), 0, 0.5, num_points)

# 03 ONE DS AND TWO NODES (EULER 1)
if False:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0,0]), np.array([0,1]), 0, 0.5, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.75, 0.5]), 
                                         np.array([0.5, 0]), 2, 0.2, num_points, 
                                         True, np.array([[1,0.5]]), 
                                         np.array([[0,1]]))
    
# 04 ONE DS AND TWO PAIRS OF NODES (ON THE WAY TO EULER 2)
if False:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0,0]), np.array([0,1]), 0, 0.5, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.75, 0.25]), 
                                         np.array([0.5, 0]), 2, 0.3, num_points, 
                                         True, np.array([[1,0.5]]), 
                                         np.array([[0,1]]))
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.75, 0.75]), 
                                         np.array([0.5, 0]), 2, 0.3, num_points, 
                                         True, np.array([[1,0.5]]), 
                                         np.array([[0,1]]))
    
# 05 REMOVING THE DS (EULER 2)
if True:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.5,0]), np.array([0,1]), 0, 0.5, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25, 0.25]), 
                                         np.array([0.5, 0]), 2, 0.3, num_points, 
                                         True, np.array([[0.5,0.25]]), 
                                         np.array([[0,1]]))
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25, 0.75]), 
                                         np.array([0.5, 0]), 2, 0.3, num_points, 
                                         True, np.array([[0.5,0.75]]), 
                                         np.array([[0,1]]))
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.5,1]), 
                                         np.array([0,-1]), 0, 0.2, num_points, 
                                         True, 
                                         np.array([[0.5, 0.25],[0.5, 0.75]]), 
                                         np.array([[1,0], [1,0]]))

#plotting the vectors
if True:
    k = np.linspace(0,1,num_points,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    u = blochvectors[:,:,0,1]
    v = blochvectors[:,:,1,1]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

# calculating Euler class
if True:
    # Defining the patch indices
    minx = int(kxmin * num_points)
    maxx = int(kxmax * num_points)
    miny = int(kymin * num_points)
    maxy = int(kymax * num_points)

    # Making the patch one index larger to calculate derivatives
    numx = maxx - minx
    numy = maxy - miny
    dkx = (maxx - minx) / (num_points * numx)
    dky = (maxy - miny) / (num_points * numy)

    patch = blochvectors[minx:maxx+1, miny:maxy+1]

    # Calculating derivatives on the patch (it is already gauge fixed)
    xder = (patch[1:] - patch[:-1]) / dkx
    yder = (patch[:,1:] - patch[:,:-1]) / dky

    # Calculating the Euler form on the patch
    Eu = np.zeros((numx,numy), dtype='float')
    for i in range(numx):
        for j in range(numy):
            Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]],yder[i,j,:,bands[1]]) 
            - np.vdot(yder[i,j,:,bands[0]],xder[i,j,:,bands[1]]))
            # removing singularities:
            if np.abs(Eu[i,j]) > divergence_threshold_surface:
                Eu[i,j] = 0

    # Plotting Euler curvature
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x,y = np.meshgrid(range(numx),range(numy),indexing='ij')
    surf1 = ax.plot_surface(x,y, Eu, cmap=cm.YlGnBu,
                                linewidth=0)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()

    Eu = Eu * dkx * dky

    # Calculating the Euler connection along the boundary
    # This consists of four segments indicated by integration direction
    right = np.zeros((numx,), dtype= 'float')
    up = np.zeros((numy,), dtype= 'float')
    left = np.zeros((numx,), dtype= 'float')
    down = np.zeros((numy,), dtype= 'float')

    for i in range(numx):
        right[i] = np.vdot(patch[i,0,:,bands[0]], xder[i,0,:,bands[1]])
        left[i] = np.vdot(patch[i,-2,:,bands[0]], 
                          xder[i,-1,:,bands[1]])
        if np.abs(right[i]) > divergence_threshold_boundary:
            right[i] = 0
        if np.abs(left[i]) > divergence_threshold_boundary:
            left[i] = 0
        
    for i in range(numy):
        up[i] = np.vdot(patch[-2,i,:,bands[0]],
                        yder[-1,i,:,bands[1]])
        down[i] = np.vdot(patch[0,i,:,bands[0]],yder[0,i,:,bands[1]])
        if np.abs(up[i]) > divergence_threshold_boundary:
            up[i] = 0
        if np.abs(down[i]) > divergence_threshold_boundary:
            down[i] = 0

    # Plotting the Euler connection
    boundary = np.concatenate([right, up , left, down])
    plt.plot(boundary)
    plt.ylim(-divergence_threshold_boundary, divergence_threshold_boundary)
    plt.show()
    plt.close()

    right = right * dkx
    left = left * -dkx
    up = up * dky
    down = down * -dky

    # Integrating the surface term
    surface_term = np.sum(Eu) / (2*np.pi)

    # Integrating the boundary term
    boundary_term = (np.sum(right[1:] + right[:-1]) 
                     + np.sum(left[1:] + left[:-1])
                     + np.sum(up[1:] + up[:-1])
                     + np.sum(down[1:] + down[:-1])) / 2 / (2*np.pi)
    
    print('Boundary term: ', boundary_term)
    print('Surface term: ', surface_term)
    chi = surface_term - boundary_term
    print('Euler class: ', chi)
    
    



