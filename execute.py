from tight_binding.solver import driven_bandstructure2D
from tight_binding.hamiltonians import driven_bloch_hamiltonian_kagome
import numpy as np

b_1 = 2*np.pi * np.array([1, -3**-0.5])
b_2 = 2*np.pi * np.array([0, 2*3**-0.5])
a_1 = np.array([1,0])
a_2 = np.array([0.5, 0.5*3**0.5])
d_ab = -0.5*a_2
d_ac = 0.5*(a_1-a_2)
d_bc = 0.5*a_1
r_a = np.array([0,0])
r_b = -d_ab
r_c = -d_ac

offsets = np.array([r_a,r_b,r_c])

def A(t):
    return np.array([2*np.cos(6*t), 0])

H = driven_bloch_hamiltonian_kagome(-0.5,1,-0.5,1,1,A)
bs = driven_bandstructure2D(H,
                            2*np.pi / 6,
                            0.01,
                            np.array([1,0]),
                            np.array([0.5, 0.5*3**0.5]),
                            3,
                            -1.2*np.pi)

bs.compute_bandstructure(100)
bs.plot_bandstructure('test.png', -1.25*np.pi, 1.25*np.pi, -1.25*np.pi, 1.25*np.pi)