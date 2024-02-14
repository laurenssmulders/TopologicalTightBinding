import numpy as np
from tight_binding.bandstructure import locate_nodes, plot_bandstructure2D
from tight_binding.topology import locate_dirac_strings

name = 'SP4_driven_Ax_Ay_w_dA_dC_1_1_11_-1.0_-3.5'
file = 'figures/square/SP4/driven/bandstructures/' + name + '/' + name + '_grids/' + name + '_grid.npy'
energies = np.load(file)

a_1 = np.array([1,0])
a_2 = np.array([0,1])

locate_nodes(energies, a_1, a_2, 'test.png', node_threshold=0.1)

plot_bandstructure2D(energies, a_1, a_2, 'test.png', lowest_quasi_energy=-np.pi / 4, bands_to_plot=[1,1,1])

