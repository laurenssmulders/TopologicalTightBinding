import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import compute_patch_euler_class, gauge_fix_grid, gauge_fix_path
from tight_binding.hamiltonians import kagome_hamiltonian_static
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.diagonalise import compute_eigenstates

H = kagome_hamiltonian_static(0,0,0,1,1)

chi = compute_patch_euler_class(-1,1,-1,1,[1,2],H,100,0,0,0,0,'static')
print(chi)