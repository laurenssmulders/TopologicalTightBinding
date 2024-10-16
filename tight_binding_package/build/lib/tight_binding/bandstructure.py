"""Functions to calculate the bandstructure for static or driven systems."""

# 01 IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .diagonalise import compute_eigenstates
from .utilities import compute_reciprocal_lattice_vectors_2D
import scipy.linalg as la

def sort_energy_path(energies, blochvectors, 
                     regime='driven'):
    """Assigns the energies and blochvectors to the right bands for a 1D path of
    energies.
    
    Parameters
    ----------
    energy_grid: numpy array
        The grid with all the energies
    blochvector_grid: numpy array
        The grid with all the blochvectors
    regime: str
        'driven' or 'static'

    Returns
    -------
    energies_sorted: numpy array
    blochvectors_sorted: numpy array
    """
    if regime == 'driven':
        for i in range(energies.shape[0]):
            if i == 0:
                ind = np.argsort(energies[i])
                energies[i] = energies[i,ind]
                rows = np.linspace(0,len(ind)-1,len(ind)).astype('int')
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
                rows = np.linspace(0,len(ind)-1,len(ind)).astype('int')
                blochvectors[i] = blochvectors[i,rows[:,np.newaxis],
                                               ind[np.newaxis,:]]
    elif regime == 'static':
        for i in range(energies.shape[0]):
            ind = np.argsort(energies[i])
            energies[i] = energies[i,ind]
            rows = np.linspace(0,len(ind)-1,len(ind)).astype('int')
            blochvectors[i] = blochvectors[i,rows[:,np.newaxis],
                                               ind[np.newaxis,:]]
    
    return energies, blochvectors

def sort_energy_grid(energy_grid,
                     blochvector_grid,
                     regime='driven'):
    """Assigns the energies and blochvectors to the right bands for a 2D grid of
    energies.
    
    Parameters
    ----------
    energy_grid: numpy array
        The grid with all the energies
    blochvector_grid: numpy array
        The grid with all the blochvectors
    regime: str
        'driven' or 'static'

    Returns
    -------
    energies_sorted: numpy array
    blochvectors_sorted: numpy array
    """
    energies_sorted = np.zeros(energy_grid.shape, dtype='float')
    blochvectors_sorted = np.zeros(blochvector_grid.shape, dtype='float')
    if regime == 'driven':
        for i in range(energy_grid.shape[0]):
            for j in range(energy_grid.shape[1]):
                if i == 0 and j == 0:
                    ind = np.argsort(energy_grid[i,j])
                    energies_sorted[i,j] = energy_grid[i,j,ind]
                    rows = np.array([0,1,2])
                    blochvectors_sorted[i,j] = blochvector_grid[i,j,
                                                            rows[:,np.newaxis],
                                                            ind[np.newaxis,:]]
                elif j == 0:
                    ind = np.argsort(energy_grid[i,j])
                    differences = np.zeros((3,), dtype='float')
                    for shift in range(3):
                        ind_roll = np.roll(ind,shift)
                        diff =((energy_grid[i,j,ind_roll]
                                 - energies_sorted[i-1,j]) % (2*np.pi))
                        diff = (diff + 2*np.pi*np.floor((-np.pi-diff) 
                                                        / (2*np.pi) + 1))
                        diff = np.abs(diff)
                        diff = np.sum(diff)
                        differences[shift] = diff
                    minimum = np.argmin(differences)
                    ind = np.roll(ind, minimum)
                    energies_sorted[i,j] = energy_grid[i,j,ind]
                    rows = np.array([0,1,2])
                    blochvectors_sorted[i,j] = blochvector_grid[i,j,
                                                            rows[:,np.newaxis],
                                                            ind[np.newaxis,:]]
                else:
                    ind = np.argsort(energy_grid[i,j])
                    differences = np.zeros((3,), dtype='float')
                    for shift in range(3):
                        ind_roll = np.roll(ind,shift)
                        diff =((energy_grid[i,j,ind_roll]
                                 - energies_sorted[i,j-1]) % (2*np.pi))
                        diff = (diff + 2*np.pi*np.floor((-np.pi-diff) 
                                                        / (2*np.pi) + 1))
                        diff = np.abs(diff)
                        diff = np.sum(diff)
                        differences[shift] = diff
                    minimum = np.argmin(differences)
                    ind = np.roll(ind, minimum)
                    energies_sorted[i,j] = energy_grid[i,j,ind]
                    rows = np.array([0,1,2])
                    blochvectors_sorted[i,j] = blochvector_grid[i,j,
                                                            rows[:,np.newaxis],
                                                            ind[np.newaxis,:]]

    elif regime == 'static':
        for i in range(energy_grid.shape[0]):
            for j in range(energy_grid.shape[1]):
                ind = np.argsort(energy_grid[i,j])
                energies_sorted[i,j] = energy_grid[i,j,ind]
                rows = np.array([0,1,2])
                blochvectors_sorted[i,j] = blochvector_grid[i,j,
                                                            rows[:,np.newaxis],
                                                            ind[np.newaxis,:]]
    
    return energies_sorted, blochvectors_sorted

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
    """Computes the bandstructure for a given bloch hamiltonian.
    
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
        A 3D array with the energies at each point. energy_grid[i,j] is an array
        of the energies at k = i / num_points * b1 + j / num_points * b2
    blochvector_grid: numpy.ndarray
        A 4D array with the blochvectors at each point. blochvector_grid[i,j] is 
        an array of the blochvectors at 
        k = i / num_points * b1 + j / num_points * b2
    """
    b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)

    # Creating a grid of coefficients for the reciprocal lattice vectors
    alpha = np.linspace(0,1,num_points,endpoint=False)
    alpha_1, alpha_2 = np.meshgrid(alpha, alpha, indexing='ij')

    # Finding the corresponding k_vectors
    kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
    ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]

    # Creating arrays for the energies and blochvectors
    if regime == 'static':
        dim = hamiltonian(np.array([0,0])).shape[0]
    elif regime == 'driven':
        dim = hamiltonian(np.array([0,0]),0).shape[0]
    energy_grid = np.zeros((num_points,num_points,dim), dtype='float')
    blochvector_grid = np.zeros((num_points,num_points,dim,dim), dtype='complex')

    # Filling the arrays
    for i in range(num_points):
        for j in range(num_points):
            k = np.array([kx[i,j],ky[i,j]])
            energies, blochvectors = compute_eigenstates(hamiltonian, k, omega, 
                                                         num_steps, 
                                                         lowest_quasi_energy, 
                                                         enforce_real, method, 
                                                         regime)
            energy_grid[i,j] = energies
            blochvector_grid[i,j] = blochvectors
    
    energies_sorted, blochvectors_sorted = sort_energy_grid(energy_grid, 
                                                            blochvector_grid, 
                                                            regime)

    return energies_sorted, blochvectors_sorted

def compute_bandstructure2D_grid(hamiltonian,
                                 omega=0,
                                 lowest_quasi_energy=-np.pi,
                                 enforce_real=True):
    """Computes Floquet bandstructure.
    
    Parameters
    ----------
    hamiltonian: 5D array
        Array where hamiltonian[t,i,j] is the bloch hamiltonian at position i,j
        and time t.
    lowest_quasi_energy: float
        The lower bound for the quasi energies
    enforce_real: bool
        Whether to ensure that blochvectors are real
        
    Returns
    -------
    energies: 3D array
    blochvectors: 4D array
    """
    num_points = hamiltonian.shape[1]
    num_steps = hamiltonian.shape[0]
    T = 2*np.pi/omega
    # Calculating the time evolution operator
    print('Calculating the time evolution operator...')
    U = np.zeros((num_points,num_points,3,3), dtype='complex')
    for i in range(num_points):
        for j in range(num_points):
            U[i,j] = np.identity(3)
    dt = T / num_steps
    for step in range(num_steps):
        U = np.matmul(la.expm(-1j*hamiltonian[step]*dt),U)

    # checking unitarity
    identity = np.zeros((num_points,num_points,3,3), dtype='complex')
    for i in range(num_points):
        for j in range(num_points):
            identity[i,j] = np.identity(3)
    error = np.sum(np.matmul(U,np.conjugate(np.transpose(U,(0,1,3,2)))) - identity) / num_points**2
    if error > 1e-5:
        print('High normalisation error!: {error}'.format(error=error))

    # diagonalising
    print('Diagonalising...')
    eigenvalues, eigenvectors = np.linalg.eig(U)
    energies = np.real(np.log(eigenvalues) / (-1j))
    errors = np.real(np.log(eigenvalues)) / num_points**2 #checking for real eigenenergies
    if np.sum(errors) > 1e-5:
        print('Imaginary quasienergies!')

    # getting the energies in the right range
    energies = (energies + 2*np.pi*np.floor((lowest_quasi_energy-energies) 
                                                    / (2*np.pi) + 1))
    blochvectors = eigenvectors

    # enforcing real blochvectors
    for i in range(num_points):
        for j in range(num_points):
            for k in range(3):
                phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,k], 
                                                blochvectors[i,j,:,k])))
                blochvectors[i,j,:,k] = np.real(blochvectors[i,j,:,k] * np.exp(-1j*phi))
                blochvectors = np.real(blochvectors)
    
    # sorting the energies and vectors
    energies, blochvectors = sort_energy_grid(energies, blochvectors)
    return energies, blochvectors

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
                         discontinuity_threshold=0.05,
                         show_plot=True,
                         r=10,
                         c=10):
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
    discontinuity_threshold: float
        The values to not plot near the upper and lower boundaries of the FBZ
    show_plot: bool
        Whether to show the plot
    r: float
        The rstride value for the plot
    c: float
        The cstride value for the plot
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
        kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
        ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]
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
    if regime == 'driven':
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if bands_to_plot[0]:
        surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.YlGnBu, edgecolor='darkblue',
                                linewidth=0, rstride=r, cstride=c)
    if bands_to_plot[1]:
        surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.PuRd, edgecolor='purple',
                                linewidth=0, rstride=r, cstride=c)
    if bands_to_plot[2]:
        surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.YlOrRd, edgecolor='darkred',
                                linewidth=0, rstride=r, cstride=c)
        
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5})

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
    ax.set_box_aspect([1, 1, 2])
    plt.savefig(save)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    if show_plot:
        plt.show()
    plt.close()

def locate_nodes(energy_grid,
                 a_1,
                 a_2,
                 save,
                 node_threshold = 0.05,
                 kxmin=-np.pi, 
                 kxmax=np.pi, 
                 kymin=-np.pi, 
                 kymax=np.pi,
                 regime='driven',
                 show_plot=True,
                 title= ''):
    """Plots the nodes in the each band gap for a given band structure.
    
    Parameters
    ----------
    energy_grid: numpy.ndarray
        The energies to find the nodes from
    a_1: numpy.ndarray
        The first lattice vector
    a_2: numpy.ndarray
        The second lattice vector
    save: str
        The place to save the plot
    node_threshold: float
        How close the bands have to be to each other for a point to be 
        considered a node
    kxmin: float
        The minimum kx value to plot
    kxmax: float
        The maximum kx value to plot
    kymin: float
        The minimum ky value to plot
    kymax: float
        The maximum ky value to plot
    regime: str
        'driven'or 'static'
    show_plot: bool
        Whether to show the plot
    title: str
        Plot title
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
        kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
        ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]
        span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
        
    # Specifying which indices in the original array correspond to indices in 
    # the extended array
    i = ((alpha_1%1) * num_points).astype(int)
    j = ((alpha_2%1) * num_points).astype(int)
    energy_grid_extended = energy_grid[i,j]
    E = np.transpose(energy_grid_extended, (2,0,1))

    if regime == 'driven':
        gap_1 = (E[1] - E[0]) % (2*np.pi)
        gap_2 = (E[2] - E[1]) % (2*np.pi)
        gap_3 = (E[0] - E[2]) % (2*np.pi)

        gap_1 = abs((gap_1 + 2*np.pi*np.floor((-np.pi - gap_1) / (2*np.pi) + 1)))
        gap_2 = abs((gap_2 + 2*np.pi*np.floor((-np.pi - gap_2) / (2*np.pi) + 1)))
        gap_3 = abs((gap_3 + 2*np.pi*np.floor((-np.pi - gap_3) / (2*np.pi) + 1)))
    else:
        print('Static regime not implemented yet.')

    gap_1 = gap_1 < node_threshold
    gap_2 = gap_2 < node_threshold
    gap_3 = gap_3 < node_threshold

    gap_1_nodes_kx = []
    gap_2_nodes_kx = []
    gap_3_nodes_kx = []
    gap_1_nodes_ky = []
    gap_2_nodes_ky = []
    gap_3_nodes_ky = []
    

    for i in range(gap_1.shape[0]):
        for j in range(gap_1.shape[1]):
            if gap_1[i,j]:
                gap_1_nodes_kx.append(kx[i,j])
                gap_1_nodes_ky.append(ky[i,j])
    
    for i in range(gap_2.shape[0]):
        for j in range(gap_2.shape[1]):
            if gap_2[i,j]:
                gap_2_nodes_kx.append(kx[i,j])
                gap_2_nodes_ky.append(ky[i,j])
    
    for i in range(gap_3.shape[0]):
        for j in range(gap_3.shape[1]):
            if gap_3[i,j]:
                gap_3_nodes_kx.append(kx[i,j])
                gap_3_nodes_ky.append(ky[i,j])

    fig = plt.figure()
    plt.scatter(gap_1_nodes_kx, gap_1_nodes_ky, label='Gap 1')
    plt.scatter(gap_2_nodes_kx, gap_2_nodes_ky, label='Gap 2')
    plt.scatter(gap_3_nodes_kx, gap_3_nodes_ky, label='Gap 3')
    plt.xlim(kxmin, kxmax)
    plt.ylim(kymin, kymax)
    ticks = [-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi]
    tick_labels = ['$-\pi$', '', '', '', '$\pi$']
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.savefig(save)
    if show_plot:
        plt.show()
    plt.close()

def compute_finite_geometry_bandstructure2D(hamiltonian, 
                            a_1, 
                            a_2,
                            cut,
                            num_points, 
                            omega=0, 
                            num_steps=0, 
                            lowest_quasi_energy=-np.pi,
                            enforce_real=True,
                            method='trotter', 
                            regime='driven'):
    """Computes the bandstructure for a given bloch hamiltonian.
    
    Parameters
    ----------
    hamiltonian: function
        The bloch hamiltonian for which to calculate the bandstructure
    a_1: numpy.ndarray
        The first eigenvector
    a_2: numpy.ndarray
        The second eigenvector
    cut: int
        The lattice vector along which to cut, 0 or 1
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
        A 2D array with the energies at each point. energy_grid[i] is an array
        of the energies at k = i / num_points * b1 or j / num_points * b2,
        depending on the cut direction.
    blochvector_grid: numpy.ndarray
        A 3D array with the blochvectors at each point. blochvector_grid[i] is 
        an array of the blochvectors at 
        k = i / num_points * b1 or j / num_points * b2
    """
    L = hamiltonian(0,0).shape[0] // 3
    b = compute_reciprocal_lattice_vectors_2D(a_1, a_2)
    k = np.linspace(-0.5*np.linalg.norm(b[cut]), 0.5*np.linalg.norm(b[cut]), num_points)
    E = np.zeros((num_points, 3*L), dtype='float')
    blochvectors = np.zeros((num_points, 3*L, 3*L), dtype='complex')

    for i in range(len(k)):
        energies, eigenvectors = compute_eigenstates(hamiltonian,k[i],omega,num_steps,
                                                    lowest_quasi_energy,False,method='Runge-Kutta')
        E[i] = energies
        blochvectors[i] = eigenvectors

    E, blochvectors = sort_energy_path(E,blochvectors)

    return E, blochvectors

def plot_finite_geometry_bandstructure2D(energies,
                         a_1,
                         a_2,
                         cut,
                         save, 
                         lowest_quasi_energy=-np.pi,
                         regime='driven',
                         show_plot=True):
    """Plots the bandstructure calculated from compute_bandstructure2D for 3 
    band systems
    
    Parameters
    ----------
    energies: numpy.ndarray
        The energies to plot
    a_1: numpy.ndarray
        The first lattice vector
    a_2: numpy.ndarray
        The second lattice vector
    cut: int
        The lattice vector along which the cut is made. 0 or 1.
    save: str
        The place to save the plot
    lowest_quasi_energy: float
        The bottom of the FBZ
    regime: str
        'driven'or 'static' (only driven implemented)
    show_plot: bool
        Whether to show the plot
    """
    # ONLY FOR DRIVEN REGIME
    b = compute_reciprocal_lattice_vectors_2D(a_1, a_2)
    num_points = energies.shape[0]
    k = np.linspace(-0.5*np.linalg.norm(b[cut]), 0.5*np.linalg.norm(b[cut]), 
                    num_points)
    plt.plot(k,energies,c='0')
    plt.ylabel('$2\pi E / \omega$')
    plt.yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
                3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                    '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
    if cut == 1:
        plt.xlabel('$k_y$')
    elif cut == 0:
        plt.xlabel('$k_x$')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
    plt.xlim(-0.5*np.linalg.norm(b[cut]), 0.5*np.linalg.norm(b[cut]))
    plt.ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
    plt.savefig(save)
    if show_plot:
        plt.show()
    plt.close()

    