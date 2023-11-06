import numpy as np

def static_bloch_hamiltonian_kagome(delta_a: float, 
                                    delta_b: float, 
                                    delta_c: float, 
                                    t: float, 
                                    a: float):
    """Defines a function for the bloch hamiltonian of the Kagome lattice. For
    unit cell sites A, B and C and tunneling parameter t.

    Parameters
    ----------
    delta_a: float
        The atomic limit potential of lattice site A.
    delta_b: float
        The atomic limit potential of lattice site B.
    delta_c: float
        The atomic limit potential of lattice site C.
    t: float
        The tunneling parameter between nearest neighbours.
    a: float
        The lattice spacing of the Bravais lattice.
    
    Returns
    -------
    H: function
        The bloch hamiltonian for the Kagome lattice.
    """

    #Defining lattice and separation vectors:
    a_1 = a*np.array([1,0])
    a_2 = a*np.array([0.5, 0.5*3**0.5])
    d_ab = -0.5*a_2
    d_ac = 0.5*(a_1-a_2)
    d_bc = 0.5*a_1

    def hamiltonian(k: np.ndarray) -> np.ndarray:
        hamiltonian = np.array([
          [delta_a, 2*t*np.cos(np.inner(k,d_ab)), 2*t*np.cos(np.inner(k,d_ac))],
          [2*t*np.cos(np.inner(k,d_ab)), delta_b, 2*t*np.cos(np.inner(k,d_bc))],
          [2*t*np.cos(np.inner(k,d_ac)), 2*t*np.cos(np.inner(k,d_bc)), delta_c]
        ])
        return hamiltonian
    
    return hamiltonian

def driven_bloch_hamiltonian_kagome(delta_a: float, 
                                    delta_b: float, 
                                    delta_c: float, 
                                    t: float, 
                                    a: float,
                                    A):
    """Defines a function for the bloch hamiltonian of the Kagome lattice. For
    unit cell sites A, B and C and tunneling parameter t.

    Parameters
    ----------
    delta_a: float
        The atomic limit potential of lattice site A.
    delta_b: float
        The atomic limit potential of lattice site B.
    delta_c: float
        The atomic limit potential of lattice site C.
    t: float
        The tunneling parameter between nearest neighbours.
    a: float
        The lattice spacing of the Bravais lattice.
    A: function
        The driving vector potential as a function of time.
    
    Returns
    -------
    H: function
        The time dependent bloch hamiltonian for the driven Kagome lattice.
    """

    #Defining lattice and separation vectors:
    a_1 = a*np.array([1,0])
    a_2 = a*np.array([0.5, 0.5*3**0.5])
    d_ab = -0.5*a_2
    d_ac = 0.5*(a_1-a_2)
    d_bc = 0.5*a_1

    def hamiltonian(k: np.ndarray, time) -> np.ndarray:
        hamiltonian = np.array([
          [delta_a, 2*t*np.cos(np.inner(k + A(time),d_ab)), 
           2*t*np.cos(np.inner(k + A(time),d_ac))],
          [2*t*np.cos(np.inner(k + A(time),d_ab)), delta_b, 
           2*t*np.cos(np.inner(k + A(time),d_bc))],
          [2*t*np.cos(np.inner(k + A(time),d_ac)), 
           2*t*np.cos(np.inner(k + A(time),d_bc)), delta_c]
        ])
        return hamiltonian
    
    return hamiltonian       