import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from tight_binding.utilities import rotate, compute_reciprocal_lattice_vectors_2D
from tight_binding.bandstructure import sort_energy_grid, plot_bandstructure2D, sort_energy_path

# Parameters
num_points = 100
T = 1
d1 = -2*np.pi/(3*T)
d2 = 0
d3 = 2*np.pi/(3*T)
N = 1
num_steps = 100
lowest_quasi_energy = -np.pi
a1 = np.array([1,0])
a2 = np.array([0,1])
L = 30

# The blochvector structures
print('Generating the blochvector structures...')
V1 = np.identity(3)
V1 = V1[np.newaxis,np.newaxis,:,:]
V1 = np.repeat(V1, num_points, 0)
V1 = np.repeat(V1, num_points, 1)

V2 = np.zeros((num_points,num_points,3,3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        V2[i,j] = rotate(np.pi*i/num_points,np.array([0,1,0]))

V3 = np.zeros((num_points,num_points,3,3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        V3[i,j] = rotate(np.pi*i/num_points,V2[i,j,:,2]) @ V2[i,j]

V4 = np.zeros((num_points,num_points,3,3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        V4[i,j] = rotate(np.pi*j/num_points,V3[i,j,:,0]) @ V3[i,j]

# Checking the zak phases of the final structure
vectors = V4[:,0]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the x direction along the middle: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

vectors = V4[:,num_points//2]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the x direction along the edge: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

vectors = V4[0,:]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the y direction along the middle: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

vectors = V4[num_points//2,:]
overlaps = np.ones((num_points, 3), dtype='complex')
for i in range(num_points):
    for band in range(3):
        overlaps[i, band] = np.vdot(vectors[i,:,band], 
                                    vectors[(i+1)%num_points,:,band])
zak_phase = np.zeros((3,), dtype='complex')
for band in range(3):
    zak_phase[band] = 1j*np.log(np.prod(overlaps[:,band]))
print('Zak phase in the y direction along the edge: {zak_phase}'.format(zak_phase=np.real(zak_phase/np.pi)))

# The hamiltonians
print('Calculating the hamiltonians...')
H1A = np.matmul(V1,np.matmul(np.diag([d1,d2,d3]),np.transpose(V1,(0,1,3,2))))
H1B = np.matmul(V1,np.matmul(np.diag([-d1-4*np.pi/T,0,-d3+4*np.pi/T]),np.transpose(V1,(0,1,3,2))))

H2A = np.matmul(V2,np.matmul(np.diag([-8*np.pi/T,2*d2,8*np.pi/T-d3]),np.transpose(V2,(0,1,3,2))))
H2B = np.matmul(V2,np.matmul(np.diag([4*np.pi/T,-d2,d3-4*np.pi/T]),np.transpose(V2,(0,1,3,2))))

H3A = np.matmul(V3,np.matmul(np.diag([-2*d1,0,3*d3]),np.transpose(V3,(0,1,3,2))))
H3B = np.matmul(V3,np.matmul(np.diag([d1,0,-d3]),np.transpose(V3,(0,1,3,2))))

H4A = np.matmul(V4,np.matmul(np.diag([d1,-3*d2,-3*d3]),np.transpose(V4,(0,1,3,2))))
H4B = np.matmul(V4,np.matmul(np.diag([0,d2,d3]),np.transpose(V4,(0,1,3,2))))

# The tunnelings
print('Calculating the tunnelings...')
n1 = np.linspace(-N,N,2*N+1,dtype='int')
n2 = np.linspace(-N,N,2*N+1,dtype='int')

J1A = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J1B = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J2A = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J2B = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J3A = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J3B = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J4A = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
J4B = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')

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

dk = 1 / num_points

for i in range(2*N+1):
    for j in range(2*N+1):
        exponent = np.exp(-1j*2*np.pi*(n1[i]*k1+n2[j]*k2))
        integrand1A = H1A * exponent
        integrand1B = H1B * exponent
        integrand2A = H2A * exponent
        integrand2B = H2B * exponent
        integrand3A = H3A * exponent
        integrand3B = H3B * exponent
        integrand4A = H4A * exponent
        integrand4B = H4B * exponent

        J1A[i,j] = -np.sum(integrand1A,(0,1)) * dk**2
        J1B[i,j] = -np.sum(integrand1B,(0,1)) * dk**2
        J2A[i,j] = -np.sum(integrand2A,(0,1)) * dk**2
        J2B[i,j] = -np.sum(integrand2B,(0,1)) * dk**2
        J3A[i,j] = -np.sum(integrand3A,(0,1)) * dk**2
        J3B[i,j] = -np.sum(integrand3B,(0,1)) * dk**2
        J4A[i,j] = -np.sum(integrand4A,(0,1)) * dk**2
        J4B[i,j] = -np.sum(integrand4B,(0,1)) * dk**2

def J(t):
    if (t%T) < T/4:
        hoppings = J1A + J1B * 8*t/T
    elif (t%T) <T/2:
        hoppings = J2A + J2B * 8*t/T
    elif (t%T) < 3*T/4:
        hoppings = J3A + J3B * 8*t/T
    else:
        hoppings = J4A + J4B * 8*t/T
    return hoppings


# Going backwards
# Calculating the hamiltonian
print('Calculating the hamiltonian backwards...')
hoppings = np.zeros((num_steps,2*N+1,2*N+1,3,3),dtype='complex')
for t in range(num_steps):
    hoppings[t] = J(t/num_steps*T)

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
            blochvectors[i,j,:,k] = np.real(blochvectors[i,j,:,k] * np.exp(-1j*phi))
blochvectors = np.real(blochvectors)

energies, blochvectors = sort_energy_grid(energies,blochvectors)

# plotting
print('Plotting...')
plot_bandstructure2D(energies,a1,a2,'test.png',lowest_quasi_energy=lowest_quasi_energy)

# Calculating Zak phases
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

# Calculating the finite hamiltonions in both directions
print('Calculating the finite hamiltonians...')
# Along a1
hamiltonian1 = np.zeros((num_steps,num_points,3*L,3*L), dtype='complex')
for t in range(num_steps):
    for m2 in range(L):
        for i in range(num_points):
            for k in range(2*N+1):
                for l in range(2*N+1):
                    if m2+n2[l] >=0 and m2+n2[l] < L:
                        hamiltonian1[t,i,3*(m2+n2[l]):3*(m2+n2[l])+3,3*m2:3*m2+3] -= J(t/num_steps*T)[k,l]*np.exp(1j*2*np.pi*n1[k]*i/num_points)

# Along a1
hamiltonian2 = np.zeros((num_steps,num_points,3*L,3*L), dtype='complex')
for t in range(num_steps):
    for m1 in range(L):
        for j in range(num_points):
            for k in range(2*N+1):
                for l in range(2*N+1):
                    if m1+n1[k] >=0 and m1+n1[k] < L:
                        hamiltonian2[t,j,3*(m1+n1[k]):3*(m1+n1[k])+3,3*m1:3*m1+3] -= J(t/num_steps*T)[k,l]*np.exp(1j*2*np.pi*n2[l]*j/num_points)


# Calculating the finite time evolution operators
print('Calculating the finite time evolution operators...')
U1 = np.identity(3*L)
U1 = U1[np.newaxis,:,:]
U1 = np.repeat(U1,num_points,0)

U2 = np.identity(3*L)
U2 = U2[np.newaxis,:,:]
U2 = np.repeat(U2,num_points,0)

dt = T / num_steps
for t in range(num_steps):
    U1 = np.matmul(la.expm(-1j*hamiltonian1[t]*dt),U1)
    U2 = np.matmul(la.expm(-1j*hamiltonian2[t]*dt),U2)

# Checking unitarity
identity = np.identity(3*L)
identity = identity[np.newaxis,:,:]
identity = np.repeat(identity,num_points,0)
error1 = np.sum(np.abs(identity - np.matmul(U1,np.conjugate(np.transpose(U1,(0,2,1))))))
if error1 > 1e-5:
    print('High normalisation error 1!: {error}'.format(error=error1))
error2 = np.sum(np.abs(identity - np.matmul(U2,np.conjugate(np.transpose(U2,(0,2,1))))))
if error2 > 1e-5:
    print('High normalisation error 2!: {error}'.format(error=error2))

# diagonalising
print('Diagonalising the finite time evolution operators...')
eigenvalues1, blochvectors1 = np.linalg.eig(U1)
eigenvalues2, blochvectors2 = np.linalg.eig(U2)

energies1 = np.real(1j*np.log(eigenvalues1))
energies1 = (energies1 + 2*np.pi*np.floor((lowest_quasi_energy-energies1) 
                                                / (2*np.pi) + 1))
energies2 = np.real(1j*np.log(eigenvalues2))
energies2 = (energies2 + 2*np.pi*np.floor((lowest_quasi_energy-energies2) 
                                                / (2*np.pi) + 1))

# enforcing reality of the blochvectors
for k in range(3):
    for i in range(num_points):
            phi = 0.5*np.imag(np.log(np.inner(blochvectors1[i,:,k], 
                                            blochvectors1[i,:,k])))
            blochvectors1[i,:,k] = np.real(blochvectors1[i,:,k] * np.exp(-1j*phi))
blochvectors1 = np.real(blochvectors1)
for k in range(3):
    for i in range(num_points):
            phi = 0.5*np.imag(np.log(np.inner(blochvectors2[i,:,k], 
                                            blochvectors2[i,:,k])))
            blochvectors2[i,:,k] = np.real(blochvectors2[i,:,k] * np.exp(-1j*phi))
blochvectors2 = np.real(blochvectors2)

energies1, blochvectors1 = sort_energy_path(energies1,blochvectors1)
energies2, blochvectors2 = sort_energy_path(energies2,blochvectors2)

# Plotting
print('Plotting the finite structures...')
b1, b2 = compute_reciprocal_lattice_vectors_2D(a1, a2)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
k1 = np.linspace(-0.5*np.linalg.norm(b1), 0.5*np.linalg.norm(b1), 
                num_points)
k2 = np.linspace(-0.5*np.linalg.norm(b2), 0.5*np.linalg.norm(b2), 
                num_points)

top = lowest_quasi_energy + 2 * np.pi
bottom = lowest_quasi_energy
for band in range(3*L):
    distance_to_top1 = np.abs(energies1[band] - top)
    distance_to_bottom1 = np.abs(energies1[band] - bottom)
    
    threshold = 0.005 * 2 * np.pi
    discontinuity_mask1 = distance_to_top1 < threshold
    energies1[band] = np.where(discontinuity_mask1, np.nan, energies1[band])
    discontinuity_mask1 = distance_to_bottom1 < threshold
    energies1[band] = np.where(discontinuity_mask1, np.nan, energies1[band])

    distance_to_top2 = np.abs(energies2[band] - top)
    distance_to_bottom2 = np.abs(energies2[band] - bottom)
    
    threshold = 0.005 * 2 * np.pi
    discontinuity_mask2 = distance_to_top2 < threshold
    energies2[band] = np.where(discontinuity_mask2, np.nan, energies2[band])
    discontinuity_mask2 = distance_to_bottom2 < threshold
    energies2[band] = np.where(discontinuity_mask2, np.nan, energies2[band])

ax1.plot(k1,energies1,c='0')
ax2.plot(k2,energies2,c='0')
ax1.set_ylabel('$2\pi E / \omega$')
ax2.set_ylabel('$2\pi E / \omega$')
ax1.set_yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
            3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
ax2.set_yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
            3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
ax1.set_xlabel = 'k_x'
ax2.set_xlabel = 'k_y'
ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
ax1.set_xlim(-0.5*np.linalg.norm(b1), 0.5*np.linalg.norm(b1))
ax2.set_xlim(-0.5*np.linalg.norm(b2), 0.5*np.linalg.norm(b2))
ax1.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
ax2.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
ax1.set_title('Cut along the x direction')
ax2.set_title('Cut along the y direction')
plt.show()
plt.close()


        







