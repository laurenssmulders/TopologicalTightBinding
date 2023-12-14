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
        dim = hamiltonian(np.transpose(np.array([[0,0]]))).shape[0]
    elif regime == 'driven':
        dim = hamiltonian(np.transpose(np.array([[0,0]])),0).shape[0]

    # Parametrizing the path
    x = np.linspace(0,1,num_points) # array what fraction of the path we're on
    d = end - start
    alpha_1 = start[0,0] + x * d[0,0]
    alpha_2 = start[1,0] + x * d[1,0]
    kx = alpha_1 * b_1[0,0] + alpha_2 * b_2[0,0]
    ky = alpha_1 * b_1[1,0] + alpha_2 * b_2[1,0]

    dk = np.transpose(np.array([kx[-1] - kx[0], ky[-1] - ky[0]]))
    diagonal = np.zeros((dim,), dtype='complex')
    for i in range(dim):
        diagonal[i] = np.exp(1j*np.vdot(dk,offsets[i]))
    offset_matrix = np.diag(diagonal)
    
    # Calculating the blochvectors along the path
    blochvectors = np.zeros((num_points,dim,dim), dtype='complex')
    for i in range(num_points):
        k = np.transpose(np.array([[kx[i],ky[i]]]))
        _, eigenvectors = compute_eigenstates(hamiltonian, k, omega, 
                                                 num_steps, lowest_quasi_energy,
                                                 enforce_real, method, regime)

        blochvectors[i] = eigenvectors

    blochvectors[-1] = np.dot(offset_matrix, blochvectors[i])

    # Calculating the Zak phases from the blochvectors
    overlaps = np.ones((num_points, dim), dtype='complex')
    for i in range(num_points - 1):
        for band in range(dim):
            overlaps[i, band] = np.vdot(blochvectors[i,:,band], 
                                        blochvectors[i+1,:,band])
    zak_phases = np.zeros((dim,), dtype='complex')
    for band in range(dim):
        zak_phases[band] = 1j*np.log(np.prod(overlaps[:,band]))
    
    return zak_phases

def compute_zak_phase_wilson_loop(hamiltonian, 
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
        dim = hamiltonian(np.transpose(np.array([[0,0]]))).shape[0]
    elif regime == 'driven':
        dim = hamiltonian(np.transpose(np.array([[0,0]])),0).shape[0]

    # Parametrizing the path
    x = np.linspace(0,1,num_points) # array what fraction of the path we're on
    d = end - start
    alpha_1 = start[0,0] + x * d[0,0]
    alpha_2 = start[1,0] + x * d[1,0]
    kx = alpha_1 * b_1[0,0] + alpha_2 * b_2[0,0]
    ky = alpha_1 * b_1[1,0] + alpha_2 * b_2[1,0]

    dk = np.transpose(np.array([kx[-1] - kx[0], ky[-1] - ky[0]]))
    diagonal = np.zeros((dim,), dtype='complex')
    for i in range(dim):
        diagonal[i] = np.exp(1j*np.vdot(dk,offsets[i]))
    offset_matrix = np.diag(diagonal)
    
    # Calculating the blochvectors along the path
    blochvectors = np.zeros((num_points,dim,dim), dtype='complex')
    for i in range(num_points):
        k = np.transpose(np.array([[kx[i],ky[i]]]))
        _, eigenvectors = compute_eigenstates(hamiltonian, k, omega, 
                                                 num_steps, lowest_quasi_energy,
                                                 enforce_real, method, regime)

        blochvectors[i] = eigenvectors

    blochvectors[-1] = np.matmul(offset_matrix, blochvectors[-1])

    # Calculating the Zak phases from the blochvectors
    overlaps = np.ones((num_points, dim, dim), dtype='complex')
    for i in range(num_points - 1):
        overlaps[i] = np.matmul(np.conjugate(np.transpose(blochvectors[i])), 
                                blochvectors[i+1])
    zak_phase = np.identity(3, dtype='complex')
    for i in range(overlaps.shape[0]):
        zak_phase = np.matmul(zak_phase,overlaps[i])
    zak_phase = 1j*la.logm(zak_phase)
    return zak_phase

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
    start_points = (np.transpose(np.array([[0,0]])) 
                    + paths * perpendicular_direction)
    end_points = start_points + direction
    zak_phases = np.zeros((3,num_lines), dtype='int')
    for i in range(num_lines):
        start = np.transpose([start_points[:,i]])
        end = np.transpose([end_points[:,i]])
        zak_phase = np.rint(np.real(compute_zak_phase(hamiltonian, a_1, a_2, 
                                                      offsets, start, end, 
                                                      num_points, omega, 
                                                      num_steps, 
                                                      lowest_quasi_energy, 
                                                      enforce_real, method, 
                                                      regime))/np.pi) % 2
        zak_phases[:,i] = zak_phase
    plt.plot(paths, zak_phases[0], label='Band 1', alpha=0.5)
    plt.plot(paths, zak_phases[1], label='Band 2', alpha=0.5)
    plt.plot(paths, zak_phases[2], label='Band 3', alpha=0.5)
    plt.xticks([0,1])
    plt.yticks([0,1], ['$0$', '$\pi$'])
    plt.legend()
    plt.show()
    plt.savefig(save)
    return zak_phases
