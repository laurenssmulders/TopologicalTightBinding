import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm

from tight_binding.bandstructure import plot_bandstructure2D, sort_energy_grid
from tight_binding.topology import gauge_fix_grid
from tight_binding.utilities import compute_reciprocal_lattice_vectors_2D

# parameters
num_points = 100
T_cut = 0.83
T = 1
d1 = -2*np.pi/(3*T)
d2 = 0
d3 = 2*np.pi/(3*T)
N = 1
num_steps = 1000
lowest_quasi_energy = -np.pi
a1 = np.array([1,0])
a2 = np.array([0,1])
r=1
c=1
discontinuity_threshold = 0

J1A = np.load('hoppings/J1A.npy')
J1B = np.load('hoppings/J1B.npy')
J2A = np.load('hoppings/J2A.npy')
J2B = np.load('hoppings/J2B.npy')
J3A = np.load('hoppings/J3A.npy')
J3B = np.load('hoppings/J3B.npy')
J4A = np.load('hoppings/J4A.npy')
J4B = np.load('hoppings/J4B.npy')

def J(t):
    if (t%T) < T/4:
        hoppings = (J1A + J1B * 8*t/T)
    elif (t%T) <T/2:
        hoppings = (J2A + J2B * 8*t/T)
    elif (t%T) < 3*T/4:
        hoppings = (J3A + J3B * 8*t/T)
    else:
        hoppings = (J4A + J4B * 8*t/T)
    return hoppings

# checking some properties
# hermiticity
diff = 0
for i in range(3):
    for j in range(3):
        diff += np.sum(np.abs(J(0.9)[i,j]-np.conjugate(np.transpose(J(0.9)[-i-1,-j-1]))))
print(diff)

#reality
diff = 0
for i in range(3):
    for j in range(3):
        diff += np.sum(np.abs(J(0.9)[i,j]-np.conjugate(J(0.9)[-i-1,-j-1])))
print(diff)

#C2 symmetry
diff = 0
for i in range(3):
    for j in range(3):
        diff += np.sum(np.abs(J(0.9)[i,j]-J(0.9)[-i-1,-j-1]))
print(diff)



# Going backwards
n1 = np.linspace(-N,N,2*N+1,dtype='int')
n2 = np.linspace(-N,N,2*N+1,dtype='int')

k1 = np.linspace(0,1,num_points,False)
k1 = k1[:,np.newaxis,np.newaxis,np.newaxis]
k1 = np.repeat(k1,num_points,1)
k1 = np.repeat(k1,3,2)
k1 = np.repeat(k1,3,3)

k2 = np.linspace(0,1,num_points,False)
k2 = k2[np.newaxis,:,np.newaxis,np.newaxis]
k2 = np.repeat(k2,num_points,0)
k2 = np.repeat(k2,3,2)
k2 = np.repeat(k2,3,3)

# Calculating the hamiltonian
print('Calculating the hamiltonian backwards...')
hoppings = np.zeros((num_steps,2*N+1,2*N+1,3,3),dtype='complex')
for t in range(num_steps):
    hoppings[t] = J(t/num_steps*T_cut)

hoppings = hoppings[:,np.newaxis,np.newaxis,:,:,:,:]
hoppings = np.repeat(hoppings,num_points,1)
hoppings = np.repeat(hoppings,num_points,2)

hamiltonian = np.zeros((num_steps,num_points,num_points,3,3),dtype='complex')
for i in range(2*N+1):
    for j in range(2*N+1):
        exponent = np.exp(1j*2*np.pi*(n1[i]*k1+n2[j]*k2))
        exponent = exponent[np.newaxis,:,:,:,:]
        exponent = np.repeat(exponent,num_steps,0)
        hamiltonian -= hoppings[:,:,:,i,j,:,:] * exponent

# Calculating the time evolution operator
print('Calculating the time evolution operator...')
U = np.identity(3)
U = U[np.newaxis,np.newaxis,:,:]
U = np.repeat(U,num_points,0)
U = np.repeat(U,num_points,1)

dt = T_cut / num_steps
for t in range(num_steps):
    U = np.matmul(la.expm(-1j*hamiltonian[t]*dt),U)

# Checking unitarity
identity = np.identity(3)
identity = identity[np.newaxis,np.newaxis,:,:]
identity = np.repeat(identity,num_points,0)
identity = np.repeat(identity,num_points,1)
error = np.sum(np.abs(identity - np.matmul(U,np.conjugate(np.transpose(U,(0,1,3,2))))))
if error > 1e-5:
    print('High normalisation error!: {error}'.format(error=error))


# Diagonalising
print('Diagonalising...')
eigenvalues, blochvectors = np.linalg.eig(U)
energies = np.real(1j*np.log(eigenvalues))
energies = (energies + 2*np.pi*np.floor((lowest_quasi_energy-energies) 
                                                / (2*np.pi) + 1))

# enforcing reality of the blochvectors
for k in range(3):
    for i in range(num_points):
        for j in range(num_points):
            phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,k], 
                                            blochvectors[i,j,:,k])))
            blochvectors[i,j,:,k] = blochvectors[i,j,:,k] * np.exp(-1j*phi)
blochvectors = np.real(blochvectors)


energies, blochvectors = sort_energy_grid(energies,blochvectors)

# checking blochvector normalisation
identity = np.identity(3)
identity = identity[np.newaxis,np.newaxis,:,:]
identity = np.repeat(identity, num_points, 0)
identity = np.repeat(identity, num_points, 1)
print(np.sum(np.abs(np.matmul(np.transpose(blochvectors,(0,1,3,2)), blochvectors)-identity)))

# Calculating Zak phases
print('Calculating Zak phases...')
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

# plotting
print('Plotting...')
# Need to periodically extend the energy array to span the whole region
b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a1, a2)
span = False
copies = 0
while not span:
    copies += 1
    alpha = np.linspace(-copies,copies,2*copies*num_points,endpoint=False)
    alpha_1, alpha_2 = np.meshgrid(alpha, alpha, indexing = 'ij')
    kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
    ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]
    span = ((np.min(kx) < -1.25*np.pi) and (np.max(kx) > 1.25*np.pi) 
                and (np.min(ky) < -1.25*np.pi) and (np.max(ky) > 1.25*np.pi))
    
# Specifying which indices in the original array correspond to indices in 
# the extended array
i = ((alpha_1%1) * num_points).astype(int)
j = ((alpha_2%1) * num_points).astype(int)
energy_grid_extended = energies[i,j]
E = np.transpose(energy_grid_extended, (2,0,1))

# Masking the data we do not want to plot
E[:, (kx>1.25*np.pi) | (kx<-1.25*np.pi) | (ky>1.25*np.pi) | (ky<-1.25*np.pi)] = np.nan

# Dealing with discontinuities
top = lowest_quasi_energy + 2 * np.pi
bottom = lowest_quasi_energy
for band in range(E.shape[0]):
    distance_to_top = np.abs(E[band] - top)
    distance_to_bottom = np.abs(E[band] - bottom)
    
    threshold = discontinuity_threshold * 2 * np.pi
    discontinuity_mask = distance_to_top < threshold
    E[band] = np.where(discontinuity_mask, np.nan, E[band])
    discontinuity_mask = distance_to_bottom < threshold
    E[band] = np.where(discontinuity_mask, np.nan, E[band])

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.YlGnBu, edgecolor='darkblue',
                        linewidth=0, rstride=r, cstride=c)
surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.PuRd, edgecolor='purple',
                        linewidth=0, rstride=r, cstride=c)
surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.YlOrRd, edgecolor='darkred',
                        linewidth=0, rstride=r, cstride=c)
    
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
ax.set_zlim(lowest_quasi_energy, np.pi)
ax.set_xlim(-1.25*np.pi,1.25*np.pi)
ax.set_ylim(-1.25*np.pi,1.25*np.pi)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_box_aspect([1, 1, 2])
plt.show()
plt.close()

