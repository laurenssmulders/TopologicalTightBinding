from tight_binding.solver import static_bandstructure2D
from tight_binding.hamiltonians import static_bloch_hamiltonian_kagome
import numpy as np

b_1 = 2*np.pi * np.array([1, -3**-0.5])
b_2 = 2*np.pi * np.array([0, 2*3**-0.5])

H = static_bloch_hamiltonian_kagome(0,0,0,1,1)
bs = static_bandstructure2D(H,
                            np.array([1,0]),
                            np.array([0.5, 0.5*3**0.5]),
                            3)

bs.compute_bandstructure(1000)
bs.plot_bandstructure('test.png', -1.5*np.pi, 1.5*np.pi, -1.5*np.pi, 1.5*np.pi)