from tight_binding2.utilitities import compute_reciprocal_lattice_vectors_2D
from tight_binding2.hamiltonians import kagome_hamiltonian_driven, kagome_hamiltonian_static
from tight_binding2.diagonalise import compute_time_evolution, compute_eigenstates
from tight_binding2.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding2.topology import compute_zak_phase
import numpy as np

a_1 = np.transpose(np.array([[1,0]]))
a_2 = np.transpose(np.array([[0.5, 0.5*3**0.5]]))
r_a = np.transpose(np.array([[0,0]]))
r_b = a_1 / 2
r_c = a_2 / 2
offsets = np.array([r_a, r_b, r_c])

H = kagome_hamiltonian_driven(0,3,-3,1,1,2,0,6,0)
energies, blochvectors = compute_bandstructure2D(H,a_1,a_2,100,6,50,-np.pi)
plot_bandstructure2D(energies, a_1, a_2, 'test.png', -1.4*np.pi, 1.4*np.pi, -1.4*np.pi, 1.4*np.pi)
