from tight_binding.utilitities import compute_reciprocal_lattice_vectors_2D
from tight_binding.hamiltonians import kagome_hamiltonian_driven, kagome_hamiltonian_static, square_hamiltonian_driven, square_hamiltonian_static
from tight_binding.diagonalise import compute_time_evolution, compute_eigenstates
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.topology import compute_zak_phase, locate_dirac_strings
import numpy as np
import matplotlib.pyplot as plt

#KAGOME
a_1 = np.transpose(np.array([[1,0]]))
a_2 = np.transpose(np.array([[0.5, 0.5*3**0.5]]))
a_3 = a_2 - a_1
r_a = a_3 / 2
r_b = a_2 / 2
r_c = a_1 / 2
offsets = np.array([r_a, r_b, r_c])

#SQUARE
#a_1 = np.transpose(np.array([[1,0]]))
#a_2 = np.transpose(np.array([[0,1]]))
#r = np.transpose(np.array([[0,0]]))
#offsets = np.array([r,r,r])

start = np.transpose(np.array([[0,0]]))
end = np.transpose(np.array([[1,0]]))

H = kagome_hamiltonian_driven(0,3,-3,1,1,2,0,6,0)

zak_phases, energies = compute_zak_phase(H,a_1,a_2,offsets,start,end,100,6,100)

print(zak_phases / np.pi)
plt.plot(energies[:,0])
plt.plot(energies[:,1])
plt.plot(energies[:,2])
plt.show()
