"""Module for calculating topological quantities"""

# 01 IMPORTS
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from .diagonalise import compute_eigenstates
from .utilitities import compute_reciprocal_lattice_vectors_2D

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
    blochvectors[-1] = np.dot(offset_matrix, blochvectors[-1])


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
                         regime='driven'):
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
        
    Returns
    -------
    """
    paths = np.linspace(0,1,num_lines)
    zak_phases = np.zeros((3,num_lines), dtype='int')
    energies = np.zeros((num_lines, num_points, 3), dtype='int')
    for i in range(num_lines):
        start = paths[i] * perpendicular_direction
        end = start + direction
        zak_phase, energy = compute_zak_phase(hamiltonian, a_1, a_2, offsets, 
                                              start, end, num_points, omega, 
                                              num_steps, lowest_quasi_energy, 
                                              enforce_real, method, regime)
        print(i)
        print(a_1,a_2,offsets,start,end,num_points,omega,num_steps)
        zak_phase = np.rint(np.real(zak_phase)/np.pi) % 2
        print(zak_phase)
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
    plt.show()
    plt.savefig(save)
    return zak_phases
