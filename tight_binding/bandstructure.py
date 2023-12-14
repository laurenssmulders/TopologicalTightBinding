"""Functions to calculate the bandstructure for static or driven systems."""

# 01 IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .diagonalise import compute_eigenstates
from .utilitities import compute_reciprocal_lattice_vectors_2D

def compute_bandstructure2D(hamiltonian, 
                            a_1, 
                            a_2, 
                            num_points, 
                            omega=0, 
                            num_steps=0, 
                            lowest_quasi_energy=-np.pi,
                            enforce_real=True,
                            method='trotter', 
                            regime='driven'):
    """Computes the bandstructure for a given static bloch hamiltonian.
    
    Parameters
    ----------
    hamiltonian: function
        The bloch hamiltonian for which to calculate the bandstructure
    a_1: numpy.ndarray
        The first eigenvector
    a_2: numpy.ndarray
        The second eigenvector
    num_points: int
        The number of points on the grid for which to calculate the eigenstates 
        along each reciprocal lattice direction
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

    Returns
    -------
    energy_grid: numpy.ndarray
        An array with the energies at each point. energy_grid[i,j] is an array
        of the energies at k = i / num_points * b1 + j / num_points * b2
    blochvector_grid: numpy.ndarray
        An array with the blochvectors at each point. blochvector_grid[i,j] is 
        an array of the blochvectors at 
        k = i / num_points * b1 + j / num_points * b2
    """
    b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)

    # Creating a grid of coefficients for the reciprocal lattice vectors
    alpha = np.linspace(0,1,num_points,endpoint=False)
    alpha_1, alpha_2 = np.meshgrid(alpha, alpha, indexing='ij')

    # Finding the corresponding k_vectors
    kx = alpha_1 * b_1[0,0] + alpha_2 * b_2[0,0]
    ky = alpha_1 * b_1[1,0] + alpha_2 * b_2[1,0]

    # Creating arrays for the energies and blochvectors
    if regime == 'static':
        dim = hamiltonian(np.transpose(np.array([[0,0]]))).shape[0]
    elif regime == 'driven':
        dim = hamiltonian(np.transpose(np.array([[0,0]])),0).shape[0]
    energy_grid = np.zeros((num_points,num_points,dim), dtype='float')
    blochvector_grid = np.zeros((num_points,num_points,dim,dim), dtype='complex')

    # Filling the arrays
    for i in range(num_points):
        for j in range(num_points):
            k = np.transpose(np.array([[kx[i,j],ky[i,j]]]))
            energies, blochvectors = compute_eigenstates(hamiltonian, k, omega, 
                                                         num_steps, 
                                                         lowest_quasi_energy, 
                                                         enforce_real, method, 
                                                         regime)
            energy_grid[i,j] = energies
            blochvector_grid[i,j] = blochvectors
    
    # Sorting the energies and blochvectors
    energies_sorted = np.zeros(energy_grid.shape, dtype='float')
    blochvectors_sorted = np.zeros(blochvector_grid.shape, dtype='complex')
    if regime == 'driven':
        for i in range(energy_grid.shape[0]):
            for j in range(energy_grid.shape[1]):
                if i == 0 and j == 0:
                    current_energies = energy_grid[i,j]
                    ind = np.argsort(current_energies)
                    energies_sorted[i,j] = energy_grid[i,j, ind]
                    blochvectors_sorted[i,j] = blochvector_grid[i,j,:,ind]
                    previous_energies = current_energies[ind]
                else:
                    current_energies = energy_grid[i,j]
                    ind = np.argsort(current_energies)
                    differences = np.zeros((3,), dtype='float')
                    for shift in range(3):
                        ind_roll = np.roll(ind,shift)
                        diff = ((current_energies[ind_roll] - previous_energies) 
                                % (2*np.pi))
                        diff = (diff + 2*np.pi*np.floor((-np.pi-diff) 
                                                        / (2*np.pi) + 1))
                        diff = np.abs(diff)
                        diff = np.sum(diff)
                        differences[shift] = diff
                    minimum = np.argmin(differences)
                    ind = np.roll(ind, minimum)
                    energies_sorted[i,j] = energy_grid[i,j, ind]
                    blochvectors_sorted[i,j] = blochvector_grid[i,j,:,ind]
                    previous_energies = energies_sorted[i,j]
    elif regime == 'static':
        for i in range(energy_grid.shape[0]):
            for j in range(energy_grid.shape[1]):
                current_energies = energy_grid[i,j]
                ind = np.argsort(current_energies)
                energies_sorted[i,j] = current_energies[ind]
                blochvectors_sorted[i,j] = blochvector_grid[i,j,:,ind]

    return energies_sorted, blochvectors_sorted

def plot_bandstructure2D(energy_grid,
                         a_1,
                         a_2,
                         save, 
                         kxmin=-1.25*np.pi, 
                         kxmax=1.25*np.pi, 
                         kymin=-1.25*np.pi, 
                         kymax=1.25*np.pi,
                         bands_to_plot=np.array([True, True, True]),
                         lowest_quasi_energy=-np.pi,
                         regime='driven',
                         discontinuity_threshold=0.05):
    """Plots the bandstructure calculated from compute_bandstructure2D for 3 
    band systems
    
    Parameters
    ----------
    energy_grid: numpy.ndarray
        The energies to plot
    a_1: numpy.ndarray
        The first lattice vector
    a_2: numpy.ndarray
        The second lattice vector
    save: str
        The place to save the plot
    kxmin: float
        The minimum kx value to plot
    kxmax: float
        The maximum kx value to plot
    kymin: float
        The minimum ky value to plot
    kymax: float
        The maximum ky value to plot
    bands_to_plot: numpy.ndarray
        Boolean array of which bands to plot
    lowest_quasi_energy: float
        The bottom of the FBZ
    regime: str
        'driven'or 'static'
    discontinuity_threshold = float
        The values to not plot near the upper and lower boundaries of the FBZ
    """
    # Need to periodically extend the energy array to span the whole region
    b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)
    num_points = energy_grid.shape[0]
    dim = energy_grid.shape[2]
    span = False
    copies = 0
    while not span:
        copies += 1
        alpha = np.linspace(-copies,copies,2*copies*num_points,endpoint=False)
        alpha_1, alpha_2 = np.meshgrid(alpha, alpha, indexing = 'ij')
        kx = alpha_1 * b_1[0,0] + alpha_2 * b_2[0,0]
        ky = alpha_1 * b_1[1,0] + alpha_2 * b_2[1,0]
        span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
    # Specifying which indices in the original array correspond to indices in 
    # the extended array
    i = ((alpha_1%1) * num_points).astype(int)
    j = ((alpha_2%1) * num_points).astype(int)
    energy_grid_extended = energy_grid[i,j]
    E = np.transpose(energy_grid_extended, (2,0,1))
    # Masking the data we do not want to plot
    E[:, (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan

    # Dealing with discontinuities
    top = lowest_quasi_energy + 2 * np.pi
    bottom = lowest_quasi_energy
    for band in range(E.shape[0]):
        distance_to_top = np.abs(E[band] - top)
        distance_to_bottom = np.abs(E[band] - bottom)
        
        threshold = discontinuity_threshold * 2 * np.pi
        discontinuity_mask = distance_to_top < threshold
        E[band] = np.where(discontinuity_mask, np.nan, E[band])
        discontinuity_mask = distance_to_bottom < threshold
        E[band] = np.where(discontinuity_mask, np.nan, E[band])

    # Plotting
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if bands_to_plot[0]:
        surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.spring,
                                linewidth=0)
    if bands_to_plot[1]:
        surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.summer,
                                linewidth=0)
    if bands_to_plot[2]:
        surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.winter,
                                linewidth=0)
    tick_values = np.linspace(-4,4,9) * np.pi / 2
    tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    if regime == 'driven':
        ax.set_zticks(tick_values)
        ax.set_zticklabels(tick_labels)
    ax.set_zlim(np.nanmin(E),np.nanmax(E))
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.savefig(save)
    plt.show()

    



    