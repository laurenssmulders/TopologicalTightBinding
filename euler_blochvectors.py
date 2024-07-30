import numpy as np
from tight_binding.utilities import rotate, cross_2D
import matplotlib.pyplot as plt
# all k vectors in terms of reciporcal lattice vector basis
num_points = 50
rot_vector = 2

# Some trivial dummy bloch vectors
blochvectors = np.identity(3)
blochvectors = blochvectors[np.newaxis,np.newaxis,:,:]
blochvectors = np.repeat(blochvectors, num_points, 0)
blochvectors = np.repeat(blochvectors, num_points, 1)

node_neg = np.array([0,0.5])
node_pos = np.array([1,0.5])
ds = np.array([0,0]) # 'p' in my notes
radius = 0.2

k = np.linspace(0,1,num_points)
kx,ky = np.meshgrid(k,k,indexing='ij')
rotation = np.zeros((num_points,num_points,3,3), dtype='float')

ds_centre = node_neg + 0.5*ds

for i in range(num_points):
    for j in range(num_points):
        k0 = np.array([kx[i,j],ky[i,j]])
        k1 = np.array([kx[i,j] + 1,ky[i,j]])
        k2 = np.array([kx[i,j] - 1,ky[i,j]])
        k3 = np.array([kx[i,j],ky[i,j] + 1])
        k4 = np.array([kx[i,j] + 1,ky[i,j] + 1])
        k5 = np.array([kx[i,j] - 1,ky[i,j] + 1])
        k6 = np.array([kx[i,j],ky[i,j] - 1])
        k7 = np.array([kx[i,j] + 1,ky[i,j] - 1])
        k8 = np.array([kx[i,j] - 1,ky[i,j] - 1])

        k_prime_options = np.array([k0-ds_centre, 
                                    k1-ds_centre, 
                                    k2-ds_centre, 
                                    k3-ds_centre, 
                                    k4-ds_centre, 
                                    k5-ds_centre, 
                                    k6-ds_centre, 
                                    k7-ds_centre, 
                                    k8-ds_centre])
        norms = np.zeros((9,),dtype='float')
        for k in range(9):
            norms[k] = np.linalg.norm(k_prime_options[k])
        
        k_prime = k_prime_options[np.argmin(norms)]
        
        # sector I
        if np.vdot(ds,k_prime) < -0.5*np.vdot(ds,ds):
            k_prime = k_prime + 0.5*ds
            if np.linalg.norm(k_prime) > radius:
                rotation[i,j] = np.identity(3)
            else:
                cosphi = -cross_2D(ds,k_prime) / (np.linalg.norm(ds)*np.linalg.norm(k_prime))
                phi = np.arccos(cosphi)
                rotation[i,j] = rotate(phi*(1-np.linalg.norm(k_prime)/radius), blochvectors[i,j,:,rot_vector])

        # sector III
        elif np.vdot(ds,k_prime) > 0.5*np.vdot(ds,ds):
            k_prime = k_prime - 0.5*ds
            if np.linalg.norm(k_prime) > radius:
                rotation[i,j] = np.identity(3)
            else:
                cosphi = -cross_2D(ds,k_prime) / (np.linalg.norm(ds)*np.linalg.norm(k_prime))
                phi = np.arccos(cosphi)
                rotation[i,j] = rotate(phi*(1-np.linalg.norm(k_prime)/radius), blochvectors[i,j,:,rot_vector])

        # sector II
        else:
            r = cross_2D(ds,k_prime) / np.linalg.norm(ds)
            if r < 0 or r > radius:
                rotation[i,j] = np.identity(3)
            else:
                rotation[i,j] = rotate(np.pi*(1-r/radius), blochvectors[i,j,:,rot_vector])

blochvectors = np.matmul(rotation, blochvectors)

#plotting the vectors
u = blochvectors[:,:,0,0]
v = blochvectors[:,:,1,0]
plt.quiver(kx,ky,u,v, width=0.001)
plt.show()

# calculating zak phases
vectors = blochvectors[:,20]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))


        

