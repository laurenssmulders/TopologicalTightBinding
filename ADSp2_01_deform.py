import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm

from tight_binding.bandstructure import plot_bandstructure2D, sort_energy_grid
from tight_binding.topology import gauge_fix_grid

# parameters
s = 0.35
num_points = 301
T = 1
d1 = -2*np.pi/(3*T)
d2 = 0
d3 = 2*np.pi/(3*T)
N = 1
num_steps = 100
lowest_quasi_energy = -np.pi/4
a1 = np.array([1,0])
a2 = np.array([0,1])

J1A = np.load('hoppings/J1A.npy')
J1B = np.load('hoppings/J1B.npy')
J2A = np.load('hoppings/J2A.npy')
J2B = np.load('hoppings/J2B.npy')
J3A = np.load('hoppings/J3A.npy')
J3B = np.load('hoppings/J3B.npy')
J4A = np.load('hoppings/J4A.npy')
J4B = np.load('hoppings/J4B.npy')

JA0 = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
JA0[N,N] = -np.diag([d1,d2,d3])

def J(t,s):
    if (t%T) < T/4:
        hoppings = JA0 + (J1A - JA0 + J1B * 8*t/T)*s
    elif (t%T) <T/2:
        hoppings = JA0 + (J2A - JA0 + J2B * 8*t/T)*s
    elif (t%T) < 3*T/4:
        hoppings = JA0 + (J3A - JA0 + J3B * 8*t/T)*s
    else:
        hoppings = JA0 + (J4A - JA0 + J4B * 8*t/T)*s
    return hoppings

# checking some properties
# hermiticity
diff = 0
for i in range(3):
    for j in range(3):
        diff += np.sum(np.abs(J(0.9,s)[i,j]-np.conjugate(np.transpose(J(0.9,s)[-i-1,-j-1]))))
print(diff)

diff = 0
for i in range(3):
    for j in range(3):
        diff += np.sum(np.abs(J(0.9,s)[i,j]-np.conjugate(J(0.9,s)[-i-1,-j-1])))
print(diff)

diff = 0
for i in range(3):
    for j in range(3):
        diff += np.sum(np.abs(J(0.9,s)[i,j]-J(0.9,s)[-i-1,-j-1]))
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
    hoppings[t] = J(t/num_steps*T,s)

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

dt = T / num_steps
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
plot_bandstructure2D(energies,a1,a2,'test.png',lowest_quasi_energy=lowest_quasi_energy)

blochvectors = gauge_fix_grid(blochvectors)
kx = np.linspace(0,1,num_points,False)
ky = np.linspace(0,1,num_points,False)
kx,ky = np.meshgrid(kx,ky,indexing='ij')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(kx, ky, blochvectors[:,:,0,1], cmap=cm.YlGnBu,
                            linewidth=0)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.grid(False)
ax.set_box_aspect([1, 1, 2])
plt.show()
plt.close()

