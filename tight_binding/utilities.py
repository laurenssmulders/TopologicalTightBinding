import numpy as np
from scipy import linalg as la
from scipy import constants

def compute_reciprocal_lattice_vectors_2D(a_1: np.ndarray, a_2: np.ndarray):
    """Calculates the two reciprocal lattice vectors for a 2D lattice.
    
    Parameters
    ----------
    a_1: numpy.ndarray
        The first lattice vector.
    a_2: numpy.ndarray
        The second lattice vector.

    Returns
    -------
    b_1: np.ndarray
        The first reciprocal lattice vector
    b_2: np.ndarray
        The second reciprocal lattice vector
    """

    b_1 = 2*np.pi / (a_1[0]*a_2[1]-a_2[0]*a_1[1]) * np.array([a_2[1], -a_2[0]])
    b_2 = 2*np.pi / (a_2[0]*a_1[1]-a_1[0]*a_2[1]) * np.array([a_1[1], -a_1[0]])

    return b_1, b_2

def compute_wigner_seitz_location_2d(dx, a, b):
    """Gives a collection of points within the wigner seitz cell.
    Breaks down for large difference in lattice vector lengths.

    Parameters
    ----------
    dx: float
        Spacing between points in the direction of the lattice vectors
    a: np.ndarray
        First lattice vector
    b: np.ndarray
        Second lattice vector
    """
    # Generating lattice points near the origin
    lattice_points = []
    for n in range(-3,3):
        for m in range(-3,3):
            lattice_points.append(n*a + m*b)
    
    # Calculating the spacings of the coefficients of the vectors
    da = dx / np.linalg.norm(a)
    db = dx / np.linalg.norm(b)
    region = []

    for i in np.linspace(-3,3,int(6/da)):
        for j in np.linspace(-3,3,int(6/db)):
            region.append(i*a + j*b)
    
    wigner_seitz = []
    for point in range(len(region)):
        distances = []
        for lattice_point in range(len(lattice_points)):
            distances.append(np.linalg.norm(point - lattice_point))
        if abs(min(distances) - np.linalg.norm(point)) < 1e-5:
            wigner_seitz.append(point)

    return wigner_seitz

def compute_time_evolution_operator(H, t, dt):
    """Computes the time evolution operator for a general hamiltonian.
    
    Parameters
    ----------
    H: function
        The (time-dependent) hamiltonian for which to calculate the time
        evolution.
    t: float
        The time at which to calculate the evolution.
    dt: float
        The time steps used in calculating the operator.

    Returns
    -------
    U: numpy.array
        The time evolution operator from H at time t.
    """
    U = np.identity(H(0).shape[0]) # U has the same dimension as H
    times = np.linspace(0,t,int(t/dt)+1)
    for i in range(len(times)):
        U = np.matmul(la.expm(-1j*H(times[i])*dt), U)
    return U
