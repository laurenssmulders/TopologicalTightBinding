"""Module for calculating topological quantities"""

# 01 IMPORTS
import numpy as np
from copy import copy
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from .diagonalise import compute_eigenstates
from .utilities import compute_reciprocal_lattice_vectors_2D, gell_mann, rotate, cross_2D
from .bandstructure import sort_energy_grid, plot_bandstructure2D

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
    for i in range(blochvectors.shape[0]):
        if i != 0:
            for band in range(blochvectors.shape[2]):
                if np.vdot(blochvectors[i-1,:,band], 
                           blochvectors[i,:,band]) < 0:
                    blochvectors[i,:,band] = -blochvectors[i,:,band]
                else:
                    blochvectors[i,:,band] = blochvectors[i,:,band]
    return blochvectors

def gauge_fix_grid(blochvectors):
    """Fixes the gauge of real blochvectors on a 2D grid.

     Parameters
    ----------
    blochvectors: numpy_array
        An array with the blochvectors at every point on the grid.
    
    Returns
    -------
    blochvectors: numpy array
        An array with the gauge fixed blochvectors.
    """
    blochvectors_gf = np.zeros(blochvectors.shape)
    for band in range(blochvectors.shape[3]):
        for i in range(blochvectors.shape[0]):
            for j in range(blochvectors.shape[1]):
                if i != 0:
                    if np.vdot(blochvectors_gf[i-1,j,:,band], 
                               blochvectors[i,j,:,band]) < 0:
                        blochvectors_gf[i,j,:,band] = - blochvectors[i,j,:,band]
                    else:
                        blochvectors_gf[i,j,:,band] = blochvectors[i,j,:,band]
                elif j != 0:
                    if np.vdot(blochvectors_gf[i,j-1,:,band], 
                               blochvectors[i,j,:,band]) < 0:
                        blochvectors_gf[i,j,:,band] = - blochvectors[i,j,:,band]
                    else:
                        blochvectors_gf[i,j,:,band] = blochvectors[i,j,:,band]
                else:
                    blochvectors_gf[i,j,:,band] = blochvectors[i,j,:,band]
    return blochvectors_gf

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
                rows = np.array([0,1,2])
                blochvectors[i] = blochvectors[i,rows[:,np.newaxis],
                                               ind[np.newaxis,:]]
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
                rows = np.array([0,1,2])
                blochvectors[i] = blochvectors[i,rows[:,np.newaxis],
                                               ind[np.newaxis,:]]
            
    elif regime == 'static':
        for i in range(energies.shape[0]):
            ind = np.argsort(energies[i])
            energies[i] = energies[i,ind]
            rows = np.array([0,1,2])
            blochvectors[i] = blochvectors[i,rows[:,np.newaxis],
                                            ind[np.newaxis,:]]

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

def compute_patch_euler_class(kxmin,
                              kxmax,
                              kymin,
                              kymax,
                              bands,
                              hamiltonian,
                              num_points=100,
                              omega=0,
                              num_steps=100,
                              lowest_quasi_energy=-np.pi,
                              method='trotter',
                              regime='driven',
                              divergence_threshold=5):
    """Calculates a patch euler class over a square patch.
    
    Parameters
    ----------
    kxmin: float
        The left boundary of the patch
    kxmax: float
        The right boundary of the patch
    kymin: float
        The bottom boundary of the patch
    kymax: float
        The top boundary of the patch
    bands: numpy.array
        The two bands over which to calculate the Euler class
    hamiltonian: function
        The bloch hamiltonian
    num_points: int
        The number of points along the path to evaluate the blochvectors at
    omega: float
        The angular frequency of the bloch hamiltonian in case of a driven 
        system
    num_steps: int
        The number of steps to use in the calculation of the time evolution
    lowest_quasi_energy: float
        The lower bound of the 2pi interval in which to give the quasi energies
    method: str
        The method for calculating the time evolution: trotter or Runge-Kutta
    regime: str
        'driven' or 'static'
    divergence_threshold: float
        There will be divergent terms in the Euler curvature and Berry 
        connections, terms above this threshold will be removed
    """
    a_1 = np.array([1,0])
    a_2 = np.array([0,1])
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
    
    # Plotting energies
    E = energy_grid
    if regime == 'driven':
        top = lowest_quasi_energy + 2 * np.pi
        bottom = lowest_quasi_energy
        for band in range(E.shape[0]):
            distance_to_top = np.abs(E[band] - top)
            distance_to_bottom = np.abs(E[band] - bottom)
            
            threshold = 0.05 * 2 * np.pi
            discontinuity_mask = distance_to_top < threshold
            E[band] = np.where(discontinuity_mask, np.nan, E[band])
            discontinuity_mask = distance_to_bottom < threshold
            E[band] = np.where(discontinuity_mask, np.nan, E[band])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(kx, ky, E[...,0], cmap=cm.YlGnBu,
                                linewidth=0)
    surf2 = ax.plot_surface(kx, ky, E[...,1], cmap=cm.PuRd,
                                linewidth=0)
    surf3 = ax.plot_surface(kx, ky, E[...,2], cmap=cm.YlOrRd,
                                linewidth=0)
    tick_values = np.linspace(-4,4,9) * np.pi / 2
    tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    if regime == 'driven':
        ztick_labels = ['$-2\pi$', '$-3\pi/2$', '$-\pi$', '$-\pi/2$', '0', 
                        '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        ax.set_zticks(tick_values)
        ax.set_zticklabels(ztick_labels)
    ax.set_zlim(np.nanmin(E),np.nanmax(E))
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()
    
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

    # calculating Euler curvature at each point
    Eu = np.zeros((num_points,num_points), dtype='float')
    for i in range(Eu.shape[0]):
        for j in range(Eu.shape[1]):
            Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]],yder[i,j,:,bands[1]])
                    - np.vdot(yder[i,j,:,bands[0]],xder[i,j,:,bands[1]]))
    
    # There will diverging derivative across Dirac strings: trying to remove them
    for i in range(Eu.shape[0]):
        for j in range(Eu.shape[1]):
            if np.abs(Eu[i,j]) > divergence_threshold:
                if j != 0:
                    Eu[i,j] = Eu[i,j-1]
                elif i != 0:
                    Eu[i,j] = Eu[i-1,j]
                else:
                    shift = 0
                    while np.abs(Eu[i,j]) > divergence_threshold and shift < 10:
                        shift += 1
                        Eu[i,j] = Eu[i + shift,j]
                if np.abs(Eu[i,j]) > divergence_threshold:
                        Eu[i,j] = 0

    # Plotting Euler Curvature
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(kx[:-1,:-1], ky[:-1,:-1], Eu, cmap=cm.YlGnBu,
                                linewidth=0)
    tick_values = np.linspace(-4,4,9) * np.pi / 2
    tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
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
        right[i] = np.vdot(blochvector_grid[i,0,:,bands[0]],xder[i,0,:,bands[1]])
        up[i] = np.vdot(blochvector_grid[-2,i,:,bands[0]],yder[-1,i,:,bands[1]])
        left[i] = - np.vdot(blochvector_grid[-2-i,-2,:,bands[0]],xder[-1-i,-1,:,bands[1]])
        down[i] = - np.vdot(blochvector_grid[0,-2-i,:,bands[0]],yder[0,-1-i,:,bands[1]])
    
    # Removing divergences here as well
    for i in range(len(right)):
        if np.abs(right[i]) > 5:
            if i != 0:
                right[i] = right[i-1]
            else:
                right[i] = right[i+5]
            if np.abs(right[i]) > 5:
                right[i] = 0
    for i in range(len(up)):
        if np.abs(up[i]) > 5:
            if i != 0:
                up[i] = up[i-1]
            else:
                up[i] = up[i+5]
            if np.abs(up[i]) > 5:
                up[i] = 0
    for i in range(len(left)):
        if np.abs(left[i]) > 5:
            if i != 0:
                left[i] = left[i-1]
            else:
                left[i] = left[i+5]
            if np.abs(left[i]) > 5:
                left[i] = 0
    for i in range(len(down)):
        if np.abs(down[i]) > 5:
            if i != 0:
                down[i] = down[i-1]
            else:
                down[i] = down[i+5]
            if np.abs(down[i]) > 5:
                down[i] = 0
    
    right = np.sum(right[1:] + right[:num_points-1]) / 2 * dkx
    up = np.sum(up[1:] + up[:num_points-1]) / 2 * dky
    left = np.sum(left[1:] + left[:num_points-1]) / 2 * dkx
    down = np.sum(down[1:] + down[:num_points-1]) / 2 * dky

    boundary_term = 1 / (2*np.pi) * (right + up + left + down)

    print('Surface Term:  ', surface_term)
    print('Boundary Term: ', boundary_term)
    chi = (surface_term - boundary_term)

    return chi
    
def impose_zak_phases_square(gamma_1_axis, 
                             gamma_1_angle,
                             gamma_2_axis,
                             gamma_2_angle,
                             V_0=np.identity(3),
                             num_points=100):
    """Constructs a blochvector structure for given zak_phases
    
    Parameters
    ----------
    gamma_1_axis: int
        The axis about which to rotate the blochvector dreibein along paths
        parallel to b1.
    gamma_1_angle: int
        The multiple of pi to rotate the blochvector dreibein by along paths
        parallel to b1.
    gamma_1_axis: int
        The axis about which to rotate the blochvector dreibein along paths
        parallel to b2.
    gamma_1_angle: int
        The multiple of pi to rotate the blochvector dreibein by along paths
        parallel to b1.
    V_0: 2D array
        The blochvectors at the origin.
    num_points: int
        The number of k points along each direction.

    Returns
    -------
    blochvectors: 4D array
        An array of shape (num_points,num_points,3,3), where
        blochvectors[i,j,:,k] is the blochvector for the band k at the point
        k = i / num_points * b1 + j / num_points * b2.
    """
    blochvectors = np.zeros((num_points,num_points,3,3)).astype('float')
    blochvectors[0,0] = V_0

    def R(phi,n):
        if n[0]**2 + n[1]**2 <= 1e-5:
                e0 = np.array([1,0,0])
                e1 = np.array([0,n[2],0])
                e2 = n
        else:
            e0 = np.array([-n[1],n[0],0]) / (n[0]**2+n[1]**2)**0.5
            e1 = (np.array([-n[0]*n[2],-n[1]*n[2],n[0]**2 + n[1]**2]) 
                  / (n[0]**2 + n[1]**2)**0.5)
            e2 = n
        S = np.zeros((3,3)).astype('float')
        S[:,0] = e0
        S[:,1] = e1
        S[:,2] = e2
        rot_z = np.array([
            [np.cos(phi),-np.sin(phi),0],
            [np.sin(phi),np.cos(phi),0],
            [0,0,1]
        ])
        rot = S @ rot_z @ np.transpose(S)
        return rot
    
    # zak phase along b1
    n = V_0[:,gamma_1_axis]
    for i in range(blochvectors.shape[0]):
        blochvectors[i,0] = R(gamma_1_angle*np.pi*i/num_points,n) @ V_0
    
    # zak phases along b2
    for i in range(blochvectors.shape[0]):
        n = blochvectors[i,0,:,gamma_2_axis]
        for j in range(blochvectors.shape[1]):
            blochvectors[i,j] = (R(gamma_2_angle*np.pi*j/num_points,n)
                                 @ blochvectors[i,0])
            
    return blochvectors

def find_tunnelings(energies,
                    blochvectors,
                    N_max,
                    period,
                    a1,
                    a2,
                    Ax,
                    Ay,
                    n_steps):
    """Fits a tight binding model to a given floquet bandstructure.
    
    Parameters
    ----------
    energies: 3D array
        Array of shape (n,n,3) where energies[i,j,k] is the quasi energy of band
        k at position i/n*b1 + j/n*b2. With b1 and b2 the reciprocal lattice
        vectors.
    blochvectors: 4D array
        Array of shape (n,n,3,3) where blochvectors[i,j,:,k] is the blochvector
        of band k at position i/n*b1 + j/n*b2. With b1 and b2 the reciprocal 
        lattice vectors.
    N_max: int
        The tunnelings J(n1,n2) are evaluated up to -N_max <= n1,n2 <= N_max
    period: float
        The period of the drive.
    a1: 1D array
        The first lattice vector.
    a2: 1D array
        The second lattice vector.
    Ax: float
        The x amplitude of the driving.
    Ay: float
        The y amplitude of the driving.
    n_steps: int
        The number of time steps for evaluation of the time integral.

    Returns
    -------
    J: 4D array
        The tunnelings for the model such that J[i,j] is the n1[i,j],n2[i,j] 
        hopping matrix.
    n1: 2D array
    n2: 2D array
    """
    n_points = energies.shape[0]
    # Defining S(k)
    S = np.zeros((n_points,n_points,3,3), dtype='float')
    for i in range(n_points):
        for j in range(n_points):
            S[i,j] = (-period*blochvectors[i,j]
                      @np.diag(energies[i,j])
                      @np.transpose(blochvectors[i,j]))
        
    
    # Defining the drive A(t)
    def A(t):
        return np.array([Ax,-Ay])*np.cos(2*np.pi*t/period)
    
    J = np.zeros((2*N_max+1,2*N_max+1,3,3), dtype='complex')
    n = np.linspace(-N_max, N_max, 2*N_max+1, dtype='int')
    n1, n2 = np.meshgrid(n,n,indexing='ij')

    for i in range(len(n)):
        for j in range(len(n)):

            # Calculating the static factor
            integrand = np.zeros(S.shape, dtype='complex')
            for l in range(n_points):
                for m in range(n_points):
                    k1 = l / n_points
                    k2 = m / n_points
                    integrand[l,m] = S[l,m] * np.exp(-2*np.pi*1j
                                                     *(n1[i,j]*k1+n2[i,j]*k2))

            dk = 1 / n_points
            static_factor = np.sum(integrand,(0,1))*dk**2

            # Calculating the floquet factor
            t = np.linspace(0,period,n_steps+1)
            dt = period / n_steps
            floquet_factor = 0j
            for m in range(n_steps):
                floquet_factor += (
                    np.exp(1j*np.vdot(A(t[m]),n1[i,j]*a1+n2[i,j]*a2)) 
                    + np.exp(1j*np.vdot(A(t[m+1]),n1[i,j]*a1+n2[i,j]*a2))) /2*dt
            
            J[i,j] = static_factor / floquet_factor
        
    return J, n1, n2
            
def dirac_string_rotation(blochvectors,
                          node_neg,
                          ds,
                          rot_vector,
                          radius,
                          num_points,
                          crossing_ds = False,
                          crossings = np.array([[0,0]]),
                          directions = np.array([[0,0]])):
    """Returns blochvectors with a new nodes combined with a dirac string imposed.
    
    Parameters
    ----------
    blochvectors: 4D array
        The original bloch vectors to introduce the new nodes for.
    node_neg: 1D array
        The location of one of the nodes.
    ds: 1D array
        The Dirac string in vector form, from node_neg to the other node
    rot_vector: int
        The index of the blochvector to rotate about
    radius: float
        The length over which to interpolate the rotation to the trivial case 
        away from the dirac string
    num_points: int
        The number of points in both directions of the blochvector grid
    crossing_ds: bool=False
        Whether a dirac string in another gap crossed the dirac string we're
        forming.
    crossing: 1D array=[0,0]
        The point of crossing
    direction: 1D array=[0,0]
        The direction of the crossing dirac string
    
    Returns
    -------
    rotated_blochvectors: 4D array
        The final blochvectors with the correct rotation.
    """
    k = np.linspace(0,1,num_points)
    kx,ky = np.meshgrid(k,k,indexing='ij')
    rotation = np.zeros((num_points,num_points,3,3), dtype='float')

    ds_centre = node_neg + 0.5*ds

    for i in range(num_points):
        for j in range(num_points):
            k0 = np.array([kx[i,j],ky[i,j]])
            k1 = np.array([kx[i,j] + 1,ky[i,j]])
            k2 = np.array([kx[i,j] - 1,ky[i,j]])
            k3 = np.array([kx[i,j],ky[i,j] + 1])
            k4 = np.array([kx[i,j] + 1,ky[i,j] + 1])
            k5 = np.array([kx[i,j] - 1,ky[i,j] + 1])
            k6 = np.array([kx[i,j],ky[i,j] - 1])
            k7 = np.array([kx[i,j] + 1,ky[i,j] - 1])
            k8 = np.array([kx[i,j] - 1,ky[i,j] - 1])

            k_prime_options = np.array([k0-ds_centre, 
                                        k1-ds_centre, 
                                        k2-ds_centre, 
                                        k3-ds_centre, 
                                        k4-ds_centre, 
                                        k5-ds_centre, 
                                        k6-ds_centre, 
                                        k7-ds_centre, 
                                        k8-ds_centre])
            norms = np.zeros((9,),dtype='float')
            for k in range(9):
                norms[k] = np.linalg.norm(k_prime_options[k])
            
            k_prime = k_prime_options[np.argmin(norms)]
            sign = 1
            if crossing_ds:
                for d in range(crossings.shape[0]):
                    sign *= np.sign(cross_2D(k_prime-(crossings[d]-ds_centre),directions[d]))
            
            # sector I
            if np.vdot(ds,k_prime) < -0.5*np.vdot(ds,ds):
                k_prime = k_prime + 0.5*ds
                if np.linalg.norm(k_prime) > radius:
                    rotation[i,j] = np.identity(3)
                else:
                    cosphi = -cross_2D(ds,k_prime) / (np.linalg.norm(ds)*np.linalg.norm(k_prime))
                    phi = np.arccos(cosphi)
                    rotation[i,j] = rotate(sign*(-np.pi/2 + phi)*(1-np.linalg.norm(k_prime)/radius), blochvectors[i,j,:,rot_vector])

            # sector III
            elif np.vdot(ds,k_prime) > 0.5*np.vdot(ds,ds):
                k_prime = k_prime - 0.5*ds
                if np.linalg.norm(k_prime) > radius:
                    rotation[i,j] = np.identity(3)
                else:
                    cosphi = -cross_2D(ds,k_prime) / (np.linalg.norm(ds)*np.linalg.norm(k_prime))
                    phi = np.arccos(cosphi)
                    rotation[i,j] = rotate(sign*(-np.pi/2 + phi)*(1-np.linalg.norm(k_prime)/radius), blochvectors[i,j,:,rot_vector])

            # sector II
            else:
                r = cross_2D(ds,k_prime) / np.linalg.norm(ds)
                if abs(r) > radius:
                    rotation[i,j] = np.identity(3)
                elif r < 0:
                    rotation[i,j] = rotate(sign*-np.pi/2*(1+r/radius), blochvectors[i,j,:,rot_vector])
                else:
                    rotation[i,j] = rotate(sign*np.pi/2*(1-r/radius), blochvectors[i,j,:,rot_vector])

    rotated_blochvectors = np.matmul(rotation, blochvectors)
    return rotated_blochvectors

def energy_difference(radius,
centres,
                      diff,
                      num_points):
    """Finds how much the energies should move towards each other for specific nodes.
    
    Parameters
    ----------
    radius: float
        The decay length of the dirac cone
    centres: 2D array
        Array of the positions of the nodes
    diff: float
        The distance between the two bands
    num_points: int
        The number of points in both directions of the blochvector grid

    Returns
    -------
    differences: 2D array
        The difference between old and new energies.
    """
    differences = np.zeros((num_points, num_points),dtype='float')
    k = np.linspace(0,1,num_points)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    for i in range(num_points):
        for j in range(num_points):
            k = np.array([kx[i,j],ky[i,j]])
            # finding the closest node
            distances = np.zeros((centres.shape[0]))
            for l in range(centres.shape[0]):
                distance = centres[l] - k
                for m in range(distance.shape[0]):
                    options = np.array([distance[m],1-abs(distance[m])])
                    distance[m] = np.min(np.abs(options))
                distances[l] = np.linalg.norm(distance)
            differences[i,j] = diff / 2 * np.exp(-distances[np.argmin(distances)]/radius)
    return differences



