"""Module for calculating topological quantities"""

# 01 IMPORTS
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from .diagonalise import compute_eigenstates
from .utilitities import compute_reciprocal_lattice_vectors_2D
from .bandstructure import sort_energy_grid

def gauge_fix_path(blochvectors):
    """Fixes the gauge of real blochvectors along a 1D path.
    
    Parameters
    ----------
    blochvectors: numpy_array
        An array with the blochvectors at every point in the path.
    
    Returns
    -------
    blochvectors_gf: numpy array
        An array with the gauge fixed blochvectors.
    """
    blochvectors_gf = np.zeros(blochvectors.shape, dtype='float')
    for i in range(blochvectors.shape[0]):
        if i == 0:
            blochvectors_gf[i] = blochvectors[i]
        else:
            for band in range(blochvectors.shape[2]):
                if np.vdot(blochvectors_gf[i-1,:,band], 
                           blochvectors[i,:,band]) < 0:
                    blochvectors_gf[i,:,band] = -blochvectors[i,:,band]
                else:
                    blochvectors_gf[i,:,band] = blochvectors[i,:,band]
    return blochvectors_gf

def gauge_fix_grid(blochvectors):
    """Fixes the gauge of real blochvectors on a 2D grid.

     Parameters
    ----------
    blochvectors: numpy_array
        An array with the blochvectors at every point on the grid.
    
    Returns
    -------
    blochvectors_gf: numpy array
        An array with the gauge fixed blochvectors.
    """
    # Dealing with discontinuities in the blochvectors due to degenerate 
    # subspaces at nodes
    for i in range(blochvectors.shape[0]):
        for j in range(blochvectors.shape[1]):
            if j != 0:
                if np.sum(np.abs(np.conjugate(np.transpose(blochvectors[i,j-1])) 
                    @ blochvectors[i,j] - np.identity(3))) > 0.5:
                    inner_products = np.diag(np.conjugate(
                        np.transpose(blochvectors[i,j-1])) 
                        @ blochvectors[i,j] - np.identity(3))
                    bool_array = inner_products < 0.9
                    degenerate_bands = np.argwhere(bool_array)
                    if len(degenerate_bands) != 2:
                        print('Not two degenerate bands!')
                    if np.vdot(blochvectors[i,j-1,:,degenerate_bands[0]], 
                               blochvectors[i,j,:,degenerate_bands[0]]) < 1e-3:
                        vector0 = blochvectors[i,j,:,degenerate_bands[1]]
                        vector1 = blochvectors[i,j,:,degenerate_bands[0]]
                        


    for i in range(blochvectors.shape[0]):
        for j in range(blochvectors.shape[1]):
            for band in range(blochvectors.shape[3]):
                if j != 0:
                    if np.vdot(blochvectors[i,j-1,:,band],
                               blochvectors[i,j,:,band]) < 0:
                        blochvectors[i,j,:,band] = - blochvectors[i,j,:,band]
                    else:
                        blochvectors[i,j,:,band] = blochvectors[i,j,:,band]
                elif i != 0:
                    if np.vdot(blochvectors[i-1,j,:,band],
                               blochvectors[i,j,:,band]) < 0:
                        blochvectors[i,j,:,band] = - blochvectors[i,j,:,band]
                    else:
                        blochvectors[i,j,:,band] = blochvectors[i,j,:,band]



    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #x = np.linspace(0,1,blochvectors.shape[0])
    #y = np.linspace(0,1,blochvectors.shape[1])
    #x,y = np.meshgrid(x,y)
    #surf1 = ax.plot_surface(x,y,blochvectors[:,:,0,1], cmap=cm.YlGnBu,
    #                            linewidth=0)
    #plt.show()
    return blochvectors

def compute_zak_phase(hamiltonian, 
                      a_1, 
                      a_2, 
                      offsets, 
                      start, 
                      end, 
                      num_points, 
                      omega=0, 
                      num_steps=0, 
                      lowest_quasi_energy=-np.pi, 
                      enforce_real=True,
                      method='trotter', 
                      regime='driven'):
    """Computes the Zak phase along a path from start to end.
    
    Parameters
    ----------
    hamiltonian: function
        The bloch hamiltonian
    a_1: numpy.ndarray
        The first lattice vector
    a_2: numpy.ndarray
        The second lattice vector
    offsets: numpy.ndarray
        The Wannier centre offsets
    start: numpy.ndarray
        The starting point in terms of the reciprocal lattice vectors
    end: numpy.ndarray
        The endpoint in terms of the reciprocal lattice vectors
    num_points: int
        The number of points along the path to evaluate the blochvectors at
    omega: float
        The angular frequency of the bloch hamiltonian in case of a driven 
        system
    num_steps: int
        The number of steps to use in the calculation of the time evolution
    lowest_quasi_energy: float
        The lower bound of the 2pi interval in which to give the quasi energies
    enforce_real: bool
        Whether or not to force the blochvectors to be real
    method: str
        The method for calculating the time evolution: trotter or Runge-Kutta
    regime: str
        'driven' or 'static'
        
    Returns
    -------
    zak_phases: np.ndarray
        The zak phases for each band from lowest to highest"""

    b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)
    if regime == 'static':
        dim = hamiltonian(np.array([0,0])).shape[0]
    elif regime == 'driven':
        dim = hamiltonian(np.array([0,0]),0).shape[0]

    # Parametrizing the path
    x = np.linspace(0,1,num_points) # array what fraction of the path we're on
    d = end - start
    alpha_1 = start[0] + x * d[0]
    alpha_2 = start[1] + x * d[1]
    kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
    ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]

    dk = np.array([kx[-1] - kx[0], ky[-1] - ky[0]])
    diagonal = np.zeros((dim,), dtype='complex')
    for i in range(dim):
        diagonal[i] = np.exp(1j*np.vdot(dk,offsets[i]))
    offset_matrix = np.diag(diagonal)
    
    # Calculating the blochvectors along the path
    blochvectors = np.zeros((num_points,dim,dim), dtype='complex')
    energies = np.zeros((num_points,dim), dtype='float')
    for i in range(num_points):
        k = np.array([kx[i],ky[i]])
        eigenenergies, eigenvectors = compute_eigenstates(hamiltonian, k, omega, 
                                                 num_steps, lowest_quasi_energy,
                                                 enforce_real, method, regime)
        #diagonal = np.zeros((dim,), dtype='complex')
        #for j in range(dim):
        #    diagonal[j] = np.exp(1j*np.vdot(k,offsets[j]))
        #    offset_matrix = np.diag(diagonal)
        #eigenvectors = np.matmul(offset_matrix, eigenvectors)
        energies[i] = eigenenergies
        blochvectors[i] = eigenvectors

    # Sorting the energies and blochvectors
    if regime == 'driven':
        for i in range(energies.shape[0]):
            if i == 0:
                ind = np.argsort(energies[i])
                energies[i] = energies[i,ind]
                blochvectors[i] = blochvectors[i][:,ind]
            else:
                ind = np.argsort(energies[i])
                differences = np.zeros((3,), dtype='float')
                for shift in range(3):
                    ind_roll = np.roll(ind,shift)
                    diff = ((energies[i,ind_roll] - energies[i-1])
                            % (2*np.pi))
                    diff = (diff + 2*np.pi*np.floor((-np.pi-diff) 
                                                    / (2*np.pi) + 1))
                    diff = np.abs(diff)
                    diff = np.sum(diff)
                    differences[shift] = diff
                minimum = np.argmin(differences)
                ind = np.roll(ind, minimum)
                energies[i] = energies[i,ind]
                blochvectors[i] = blochvectors[i][:,ind]
    elif regime == 'static':
        for i in range(energies.shape[0]):
            ind = np.argsort(energies[i])
            energies[i] = energies[i,ind]
            blochvectors[i] = blochvectors[i][:,ind]

    # Taking care of centre offsets
    blochvectors[-1] = np.matmul(offset_matrix, blochvectors[0])


    # Calculating the Zak phases from the blochvectors
    overlaps = np.ones((num_points - 1, dim), dtype='complex')
    for i in range(num_points - 1):
        for band in range(dim):
            overlaps[i, band] = np.vdot(blochvectors[i,:,band], 
                                        blochvectors[i+1,:,band])
    zak_phases = np.zeros((dim,), dtype='complex')
    for band in range(dim):
        zak_phases[band] = 1j*np.log(np.prod(overlaps[:,band]))

    return zak_phases, energies

def locate_dirac_strings(hamiltonian,
                         direction,
                         perpendicular_direction,
                         num_lines,
                         save,
                         a_1, 
                         a_2, 
                         offsets, 
                         num_points, 
                         omega=0, 
                         num_steps=0, 
                         lowest_quasi_energy=-np.pi, 
                         enforce_real=True,
                         method='trotter', 
                         regime='driven',
                         show_plot=True):
    """Computes the Zak phase along several parallel paths and plots them.
    
    Parameters
    ----------
    hamiltonian: function
        The bloch hamiltonian
    direction: numpy.ndarray
        The direction (and length) of the paths in terms of b1 and b2
    perpendicular_direction: numpy.ndarray
        The perpendicular direction to the path in terms of b1 and b2 and its
        length determines the range of paths drawn. It does not actually have to 
        be perpendicular.
    num_lines: int
        The number of parallel paths to draw
    save: str
        Where to save the final plot
    a_1: numpy.ndarray
        The first lattice vector
    a_2: numpy.ndarray
        The second lattice vector
    offsets: numpy.ndarray
        The Wannier centre offsets
    start: numpy.ndarray
        The starting point in terms of the reciprocal lattice vectors
    end: numpy.ndarray
        The endpoint in terms of the reciprocal lattice vectors
    num_points: int
        The number of points along the path to evaluate the blochvectors at
    omega: float
        The angular frequency of the bloch hamiltonian in case of a driven 
        system
    num_steps: int
        The number of steps to use in the calculation of the time evolution
    lowest_quasi_energy: float
        The lower bound of the 2pi interval in which to give the quasi energies
    enforce_real: bool
        Whether or not to force the blochvectors to be real
    method: str
        The method for calculating the time evolution: trotter or Runge-Kutta
    regime: str
        'driven' or 'static'
    show_plot: bool
        Whether or not to show the plot.
        
    Returns
    -------
    """
    paths = np.linspace(0,1,num_lines,endpoint=False)
    zak_phases = np.zeros((3,num_lines), dtype='int')
    energies = np.zeros((num_lines, num_points, 3), dtype='int')
    for i in range(num_lines):
        start = paths[i] * perpendicular_direction
        end = start + direction
        zak_phase, energy = compute_zak_phase(hamiltonian, a_1, a_2, offsets, 
                                              start, end, num_points, omega, 
                                              num_steps, lowest_quasi_energy, 
                                              enforce_real, method, regime)
        zak_phase = np.rint(np.real(zak_phase)/np.pi) % 2
        zak_phases[:,i] = zak_phase
        energies[i] = energy
    
    #Need to make sure Zak phases belong to the right bands
    for i in range(1,num_lines):
        differences = np.zeros((3,))
        for shift in range(3):
            ind_roll = np.roll(np.array([0,1,2]),shift)
            diff = ((energies[i][:,ind_roll] - energies[i-1])
                    % (2*np.pi))
            diff = (diff + 2*np.pi*np.floor((-np.pi-diff) 
                                            / (2*np.pi) + 1))
            diff = np.abs(diff)
            diff = np.sum(diff)
            differences[shift] = diff
        minimum = np.argmin(differences)
        ind = np.roll(np.array([0,1,2]), minimum)
        energies[i] = energies[i][:,ind]
        zak_phases[:,i] = zak_phases[ind,i]


    plt.plot(paths, zak_phases[0], label='Band 1', alpha=0.5)
    plt.plot(paths, zak_phases[1], label='Band 2', alpha=0.5)
    plt.plot(paths, zak_phases[2], label='Band 3', alpha=0.5)
    plt.xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    plt.yticks([0,1], ['$0$', '$\pi$'])
    plt.legend()
    if show_plot:
        plt.show()
    plt.savefig(save)
    plt.close()

def compute_patch_euler_class(kxmin,kxmax,kymin,kymax,bands,hamiltonian,num_points,omega,num_steps,lowest_quasi_energy,method,regime):
    kx = np.linspace(kxmin,kxmax,num_points)
    dkx = kx[1] - kx[0]
    kx_extended = np.zeros(kx.shape[0] + 1)
    kx_extended[:-1] = kx
    kx_extended[-1] = kx[-1] + dkx
    kx = kx_extended
    
    ky = np.linspace(kymin,kymax,num_points)
    dky = ky[1] - ky[0]
    ky_extended = np.zeros(ky.shape[0] + 1)
    ky_extended[:-1] = ky
    ky_extended[-1] = ky[-1] + dky
    ky = ky_extended

    kx, ky = np.meshgrid(kx,ky,indexing='ij')

    blochvector_grid = np.zeros((num_points + 1, num_points + 1, 3, 3), 
                                dtype='float')
    energy_grid = np.zeros((num_points + 1, num_points + 1, 3), dtype='float')

    # calculating the bandstructure on the patch
    for i in range(energy_grid.shape[0]):
        for j in range(energy_grid.shape[1]):
            k = np.array([kx[i,j], ky[i,j]])
            energies, blochvectors = compute_eigenstates(hamiltonian, k, omega,
                                                         num_steps,
                                                         lowest_quasi_energy,
                                                         True, method, regime)
            energy_grid[i,j] = energies
            blochvector_grid[i,j] = blochvectors
    energy_grid, blochvector_grid = sort_energy_grid(energy_grid, 
                                                     blochvector_grid, regime)
    
    # gauge fixing
    blochvector_grid = gauge_fix_grid(blochvector_grid)
    
    # calculating the x and y derivatives of the blochvectors (multiplied by dk)
    xder = np.zeros((num_points,num_points,3,3), dtype='float')
    for i in range(xder.shape[0]):
        for j in range(xder.shape[1]):
            xder[i,j] = (blochvector_grid[i+1,j] - blochvector_grid[i,j]) / dkx
    
    yder = np.zeros((num_points,num_points,3,3), dtype='float')
    for i in range(yder.shape[0]):
        for j in range(yder.shape[1]):
            yder[i,j] = (blochvector_grid[i,j+1] - blochvector_grid[i,j]) / dky
    
    # calculating Euler curvature at each point and adding up
    Eu = np.zeros((num_points,num_points), dtype='float')
    for i in range(Eu.shape[0]):
        for j in range(Eu.shape[1]):
            Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]],yder[i,j,:,bands[1]])
                    - np.vdot(yder[i,j,:,bands[0]],xder[i,j,:,bands[1]]))
            
    mask = np.abs(Eu) > 1000
    Eu = np.where(mask, np.nan, Eu)

    # Plotting Euler curvature:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(kx[:-1,:-1], ky[:-1,:-1], Eu, cmap=cm.YlGnBu, linewidth=0)
    #ax.set_zlim(-5,5)
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()

    
    
    # Gauge fixing is not perfect so there might be divergent terms
    # trying to get rid of them:
    for i in range(Eu.shape[0]):
        for j in range(Eu.shape[1]):
            if Eu[i,j] > 5:
                if j != 0:
                    Eu[i,j] = Eu[i,j-1]
                elif i != 0:
                    Eu[i,j] = Eu[i-1,j]
                else:
                    shift = 0
                    while Eu[i,j] > 5:
                        shift += 1
                        Eu[i,j] = Eu[i + shift,j]

    # Plotting Euler curvature:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(kx[:-1,:-1], ky[:-1,:-1], Eu, cmap=cm.YlGnBu, linewidth=0)
    #ax.set_zlim(-5,5)
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()

    # Doing y integrals
    integ_x = np.zeros(Eu.shape[0])
    for i in range(len(integ_x)):
        integ_x[i] = np.sum((Eu[i,1:] + Eu[i,:num_points-1]) / 2)*dky
    
    # Doing the x integral
    surface_term = 1 / (2*np.pi) * np.sum((integ_x[1:] 
                           + integ_x[:num_points-1]) / 2)*dkx

    # calculating the boundary term, dividing the boundary into 4 legs;
    # right, up, left, down
    right = np.zeros(num_points, dtype='float')
    up = np.zeros(num_points, dtype='float')
    left = np.zeros(num_points, dtype='float')
    down = np.zeros(num_points, dtype='float')

    for i in range(num_points):
        right[i] = np.vdot(blochvector_grid[i,0,:,0],xder[i,0,:,1])
        up[i] = np.vdot(blochvector_grid[-2,i,:,0],yder[-1,i,:,1])
        left[i] = - np.vdot(blochvector_grid[-2-i,-2,:,0],xder[-1-i,-1,:,1])
        down[i] = - np.vdot(blochvector_grid[0,-2-i,:,0],yder[0,-1-i,:,1])
    
    right = np.sum(right[1:] + right[:num_points-1]) / 2 * dkx
    up = np.sum(up[1:] + up[:num_points-1]) / 2 * dky
    left = np.sum(left[1:] + left[:num_points-1]) / 2 * dkx
    down = np.sum(down[1:] + down[:num_points-1]) / 2 * dky
    boundary_term = 1 / (2*np.pi) * (right + up + left + down)

    print(surface_term)
    print(boundary_term)
    chi = (surface_term - boundary_term)

    return chi
    

