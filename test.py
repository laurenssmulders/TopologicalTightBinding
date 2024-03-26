import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import impose_zak_phases_square

blochvectors = impose_zak_phases_square(2,5,0,0)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
kx = np.linspace(0,1,100)
ky = np.linspace(0,1,100)
kx, ky = np.meshgrid(kx,ky,indexing='ij')

xpath = blochvectors[:,8]
ypath = blochvectors[52,:]

overlaps = np.ones((xpath.shape[0], 3), dtype='complex')
for i in range(xpath.shape[0] - 1):
    for band in range(3):
        overlaps[i, band] = np.vdot(xpath[i,:,band], 
                                    xpath[i+1,:,band])
for band in range(3):
    overlaps[-1, band] = np.vdot(xpath[-1,:,band], 
                                    xpath[0,:,band])
    
zak_phases = np.zeros((3,), dtype='complex')
for band in range(3):
    print(np.prod(overlaps[:,band]))
    zak_phases[band] = 1j*np.log(np.prod(overlaps[:,band]))

print('Zak phase 1: ',np.real(zak_phases) / np.pi)

overlaps = np.ones((ypath.shape[0], 3), dtype='complex')
for i in range(ypath.shape[0] - 1):
    for band in range(3):
        overlaps[i, band] = np.vdot(ypath[i,:,band], 
                                    ypath[i+1,:,band])
for band in range(3):
    overlaps[-1, band] = np.vdot(ypath[-1,:,band], 
                                    ypath[0,:,band])
    
zak_phases = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phases[band] = 1j*np.log(np.prod(overlaps[:,band]))

print('Zak phase 2: ', np.real(zak_phases) / np.pi)

surf = ax.plot_surface(kx, ky, blochvectors[:,:,0,1], cmap=cm.YlGnBu,
                       linewidth=0)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.grid(False)
ax.set_box_aspect([1, 1, 2])
plt.show()
plt.close()



