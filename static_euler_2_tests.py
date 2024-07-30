import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import dirac_string_rotation, energy_difference, gauge_fix_grid
from tight_binding.bandstructure import sort_energy_grid, sort_energy_path
from tight_binding.utilities import rotate

num_points = 100
N = 4
r = 1
c = 1
divergence_threshold = 5
bands = [1,2]
kxmin = 0
kxmax = 1
kymin = 0
kymax = 1
L=100

# 01 TWO NODES AND NOTHING ELSE
blochvectors = np.identity(3)
blochvectors = blochvectors[np.newaxis,np.newaxis,:,:]
blochvectors = np.repeat(blochvectors, num_points, 0)
blochvectors = np.repeat(blochvectors, num_points, 1)

blochvectors = dirac_string_rotation(blochvectors, 0.25, 0.75, 2, 0.3, num_points)

#plotting the vectors
if True:
    k = np.linspace(0,1,num_points,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    u = blochvectors[:,:,0,0]
    v = blochvectors[:,:,1,0]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

dk = 1 / num_points
patch = blochvectors[int(kxmin * num_points)]

