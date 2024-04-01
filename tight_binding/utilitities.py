"""A module for general useful functions"""

# 01 IMPORTS
import numpy as np

# 02 VECTOR ALGEBRA
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


def gell_mann():
    """Returns a list of Gell-Mann matrices + the identity"""
    l_0 = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    l_1 = np.array([
        [0,1,0],
        [1,0,0],
        [0,0,0]
    ])

    l_2 = np.array([
        [0,-1j,0],
        [1j,0,0],
        [0,0,0]
    ])

    l_3 = np.array([
        [1,0,0],
        [0,-1,0],
        [0,0,0]
    ])

    l_4 = np.array([
        [0,0,1],
        [0,0,0],
        [1,0,0]
    ])

    l_5 = np.array([
        [0,0,-1j],
        [0,0,0],
        [1j,0,0]
    ])

    l_6 = np.array([
        [0,0,0],
        [0,0,1],
        [0,1,0]
    ])

    l_7 = np.array([
        [0,0,0],
        [0,0,-1j],
        [0,1j,0]
    ])

    l_8 = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,-2]
    ]) / 3**0.5

    return [l_0,l_1,l_2,l_3,l_4,l_5,l_6,l_7,l_8]
