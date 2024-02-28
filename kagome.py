import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import compute_patch_euler_class, gauge_fix_grid, gauge_fix_path
from tight_binding.hamiltonians import kagome_hamiltonian_driven, square_hamiltonian_static
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.diagonalise import compute_eigenstates
from tight_binding.utilitities import compute_reciprocal_lattice_vectors_2D

H = square_hamiltonian_static(0,0,0,J_ab_0=1,J_ac_0=1,J_bc_0=1,J_ac_1x=1,J_bc_1y=1,J_ab_2m=1)

chi = compute_patch_euler_class(-1,1,-1,1,[1,2],H,regime='static',divergence_threshold=1)
print(chi)