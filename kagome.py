import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import compute_patch_euler_class, gauge_fix_grid
from tight_binding.hamiltonians import kagome_hamiltonian_static
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D

H = kagome_hamiltonian_static(0,0,0,1,1)

a_1 = np.array([1,0])
a_2 = 0.5*np.array([1,3**0.5])

energies, blochvectors = compute_bandstructure2D(H,a_1,a_2,100,regime='static')

plot_bandstructure2D(energies, a_1, a_2, 'test.png', regime='static')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.linspace(0,1,blochvectors.shape[0])
y = np.linspace(0,1,blochvectors.shape[1])
x,y = np.meshgrid(x,y)
surf1 = ax.plot_surface(x,y,blochvectors[:,:,0,1], cmap=cm.YlGnBu,
                            linewidth=0)
plt.show()

blochvectors = gauge_fix_grid(blochvectors)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.linspace(0,1,blochvectors.shape[0])
y = np.linspace(0,1,blochvectors.shape[1])
x,y = np.meshgrid(x,y)
surf1 = ax.plot_surface(x,y,blochvectors[:,:,0,1], cmap=cm.YlGnBu,
                            linewidth=0)
plt.show()


#chi = compute_patch_euler_class(-0.1,0.1,-0.1,0.1,[0,1],H,100,0,0,0,0,'static')
#print(chi)