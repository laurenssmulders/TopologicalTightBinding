from tight_binding.solver import driven_bandstructure2D
from tight_binding.hamiltonians import driven_bloch_hamiltonian_kagome
import numpy as np

def A(t):
    return np.array([2*np.cos(6*t), 0])

H = driven_bloch_hamiltonian_kagome(1,1,1,1,1,A)
bs = driven_bandstructure2D(H, 
                            2*np.pi / 6, 
                            0.01, 
                            np.array([1,0]),
                            np.array([0.5,0.5*3**0.5]), 
                            50, 
                            3, 
                            1)

bs.compute_bandstructure()
bs.plot_bandstructure('driven_bandstructure2.png')