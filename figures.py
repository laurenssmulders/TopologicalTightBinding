import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

E1 = -np.pi/2
E2 = 0
E3 = 0
energies1 = np.ones((100,100))*E1
energies2 = np.ones((100,100))*E2
energies3 = np.ones((100,100))*E3

kx = np.linspace(-1.25*np.pi, 1.25*np.pi, 100)
ky = np.linspace(-1.25*np.pi, 1.25*np.pi, 100)
kx, ky = np.meshgrid(kx,ky,indexing='ij')

E = np.array([energies1,energies2,energies3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.YlGnBu, edgecolor='darkblue',
                        linewidth=0)
surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.PuRd, edgecolor='purple',
                        linewidth=0)
surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.YlOrRd, edgecolor='darkred',
                        linewidth=0)
    
# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
ax.zaxis._axinfo["grid"].update({"linewidth":0.5})

tick_values = np.linspace(-4,4,9) * np.pi / 2
tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
ax.set_xticks(tick_values)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_values)
ax.set_yticklabels(tick_labels)
ztick_labels = ['$-2\pi$', '$-3\pi/2$', '$-\pi$', '$-\pi/2$', '0', 
                '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
ax.set_zticks(tick_values)
ax.set_zticklabels(ztick_labels)
ax.set_zlim(-np.pi, np.pi)
ax.set_xlim(-1.25*np.pi,1.25*np.pi)
ax.set_ylim(-1.25*np.pi,1.25*np.pi)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$\epsilon t$')
ax.set_box_aspect([1, 1, 2])
plt.show()
plt.close()