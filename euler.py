import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import gauge_fix_grid

num_points = 100
blochvectors = np.load('blochvectors.npy')
bands = [1,2]
divergence_threshold = 1.5

# CALCULATING EULER CLASS
blochvectors = gauge_fix_grid(blochvectors)
dk = 2*np.pi / (num_points - 1)
xder = (blochvectors[1:] - blochvectors[:-1]) / dk
yder = (blochvectors[:,1:] - blochvectors[:,:-1]) / dk
Eu = np.zeros((num_points - 1, num_points - 1), dtype='float')
for i in range(num_points - 1):
    for j in range(num_points - 1):
        Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]], yder[i,j,:,bands[1]])
                   - np.vdot(yder[i,j,:,bands[0]], xder[i,j,:,bands[1]]))
        if np.abs(Eu[i,j]) > divergence_threshold:
            Eu[i,j] = 0

# Plotting Euler curvature
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x,y = np.meshgrid(range(num_points-1),range(num_points-1),indexing='ij')
surf1 = ax.plot_surface(x,y, Eu, cmap=cm.YlGnBu,
                            linewidth=0)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.grid(False)
ax.set_box_aspect([1, 1, 2])
plt.show()
plt.close()

Eu = Eu * dk**2
chi = np.sum(Eu) / (2*np.pi)
print('Euler Class: ', chi)