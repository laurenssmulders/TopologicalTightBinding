from tight_binding.solver import static_bandstructure2D
from tight_binding.hamiltonians import static_bloch_hamiltonian_kagome
import numpy as np

H = static_bloch_hamiltonian_kagome(0,0,0,-1,1)
bs = static_bandstructure2D(H,
                            np.array([1,0]),
                            np.array([0.5, 0.5*3**0.5]),
                            500,
                            3)
bs.compute_bandstructure()
bs.plot_bandstructure('test.png', -1.25*np.pi, 1.25*np.pi, -1.25*np.pi, 1.25*np.pi)