import numpy as np

blochvectors = np.load('blochvectors.npy')
num_points = blochvectors.shape[0]

vectors = blochvectors[:,0]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the x direction along the middle: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

vectors = blochvectors[:,num_points//2]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the x direction along the edge: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

vectors = blochvectors[0,:]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the y direction along the middle: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

vectors = blochvectors[num_points//2,:]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the y direction along the edge: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))