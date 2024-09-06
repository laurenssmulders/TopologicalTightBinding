import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import gauge_fix_grid

NUM_POINTS = 100
blochvectors = np.load('blochvectors.npy')
bands = [0,1]
divergence_threshold = 2

# CALCULATING EULER CLASS
blochvectors = gauge_fix_grid(blochvectors)
dk = 2*np.pi / (NUM_POINTS - 1)
xder = (blochvectors[1:] - blochvectors[:-1]) / dk
yder = (blochvectors[:,1:] - blochvectors[:,:-1]) / dk
Eu = np.zeros((NUM_POINTS - 1, NUM_POINTS - 1), dtype='float')
for i in range(NUM_POINTS - 1):
    for j in range(NUM_POINTS - 1):
        Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]], yder[i,j,:,bands[1]])
                   - np.vdot(yder[i,j,:,bands[0]], xder[i,j,:,bands[1]]))
        if np.abs(Eu[i,j]) > divergence_threshold:
            Eu[i,j] = 0

# Plotting Euler curvature
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x,y = np.meshgrid(range(NUM_POINTS-1),range(NUM_POINTS-1),indexing='ij')
surf1 = ax.plot_surface(x,y, Eu, cmap=cm.YlGnBu,
                            linewidth=0)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.grid(False)
ax.set_box_aspect([1, 1, 2])
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
plt.close()

Eu = Eu * dk**2
chi = np.sum(Eu) / (2*np.pi)
print('Euler Class: ', chi)