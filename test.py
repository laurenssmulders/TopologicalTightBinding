import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import gauge_fix_grid
from tight_binding.bandstructure import compute_bandstructure2D_grid, plot_bandstructure2D

a1 = np.array([1,0])
a2 = np.array([0,1])

hamiltonian = np.load('hamiltonian.npy')

energies, blochvectors = compute_bandstructure2D_grid(hamiltonian,2*np.pi)

plot_bandstructure2D(energies,a1,a2,'test.png')

