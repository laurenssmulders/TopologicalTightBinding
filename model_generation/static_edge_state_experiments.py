import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg as la
import copy
from tight_binding.topology import dirac_string_rotation, energy_difference, sort_energy_grid, gauge_fix_grid
from tight_binding.bandstructure import plot_bandstructure2D

NUM_POINTS = 100
N = 2
a1 = np.array([1,0])
a2 = np.array([0,1])
r=1
c=1

# 01 FINDING ENERGIES AND BLOCHVECTORS FOR EVERY STEP
# VECTORS
## starting with trivial blochvectors
trivial_vectors = np.identity(3)
trivial_vectors = trivial_vectors[np.newaxis, np.newaxis, :, :]
trivial_vectors = np.repeat(trivial_vectors, NUM_POINTS, 0)
trivial_vectors = np.repeat(trivial_vectors, NUM_POINTS, 1)

## creating two pairs of nodes in gap 1
V = trivial_vectors
V = dirac_string_rotation(V, np.array([0,0.5]), np.array([1,0]), 2, 0.5, 
                          NUM_POINTS)
V = dirac_string_rotation(V, np.array([0,0.5]), np.array([1,0]), 2, 0.5, 
                          NUM_POINTS)
V = dirac_string_rotation(V, np.array([0.5, 0]), np.array([0,1]), 0, 0.5, 
                          NUM_POINTS, True, np.array([[0.5,0.5], [0.5,0.5]]), 
                          np.array([[1,0], [1,0]]))
#V = dirac_string_rotation(V, np.array([0,0.1]), np.array([0,0.8]), 0, 0.25, NUM_POINTS)


# ENERGIES
trivial_energies = np.array([-2*np.pi/3, 0, 2*np.pi/3])
trivial_energies = trivial_energies[np.newaxis, np.newaxis, :]
trivial_energies = np.repeat(trivial_energies, NUM_POINTS, 0)
trivial_energies = np.repeat(trivial_energies, NUM_POINTS, 1)

E = np.zeros((NUM_POINTS, NUM_POINTS, 3), dtype='float')
E[:,:,0] = trivial_energies[:,:,0]
E[:,:,1] = trivial_energies[:,:,1]
E[:,:,2] = trivial_energies[:,:,2]

if False:
    differences = energy_difference(0.2, np.array([[0.25,0.25], [0.75, 0.25], [0.25,0.75], [0.75, 0.75]]), 
                                                2*np.pi/3, NUM_POINTS)

    E = np.zeros((NUM_POINTS, NUM_POINTS, 3), dtype='float')
    E[:,:,0] = trivial_energies[:,:,0] + differences
    E[:,:,1] = trivial_energies[:,:,1] - differences
    E[:,:,2] = trivial_energies[:,:,2]

if False:
    differences = energy_difference(0.3, np.array([[0.5,0]]), 2*np.pi/3, NUM_POINTS) / 2
    E[:,:,1] = E[:,:,1] + differences
    E[:,:,2] = E[:,:,2] - differences

# PLOTTING EVERYTHING FOR CHECKS
if False:
    plot_bandstructure2D(E, a1, a2, 'test.png', regime='static',r=1,c=1)
    k = np.linspace(0,1,NUM_POINTS,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')

    u = V[:,:,1,2]
    v = V[:,:,2,2]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

E_diagonal = np.zeros((NUM_POINTS, NUM_POINTS, 3, 3), dtype='float')
for i in range(NUM_POINTS):
    for j in range(NUM_POINTS):
        E_diagonal[i,j] = np.diag(E[i,j])
E = E_diagonal


# FINDING THE HAMILTONIAN
H = np.matmul(V, np.matmul(E, np.transpose(V, (0,1,3,2))))

## Checking hermiticity
hermiticity_error = (np.sum(np.abs(H 
                                  - np.conjugate(np.transpose(H, 
                                                              (0,1,3,2))))) 
                                                              / NUM_POINTS**2)
print('Hermiticity error: ', hermiticity_error)


# FINDING THE HOPPINGS
k = np.linspace(-np.pi, np.pi, NUM_POINTS)
kx, ky = np.meshgrid(k,k,indexing='ij')
dk = 2*np.pi / (NUM_POINTS - 1)
J = np.zeros((2*N+1,2*N+1,3,3), dtype='complex')
for n1 in np.linspace(-N,N,2*N+1,endpoint=True, dtype='int'):
    for n2 in np.linspace(-N,N,2*N+1,endpoint=True, dtype='int'):
        exponent = -1j * (n1*kx + n2*ky)
        exponent = np.exp(exponent)
        exponent = exponent[:,:,np.newaxis, np.newaxis]
        exponent = np.repeat(exponent, 3, 2)
        exponent = np.repeat(exponent, 3, 3)
        integrand = H * exponent
        J[n1 + N, n2 + N] =-(np.sum(integrand[:-1,:-1], (0,1))
                               *dk**2/(4*np.pi**2))
        
np.save('J.npy', J)

# GOING BACKWARDS
# CALCULATING THE HAMILTONIAN
print('Calculating the hamiltonian backwards...')
k = np.linspace(0, 2*np.pi, NUM_POINTS)
kx, ky = np.meshgrid(k,k,indexing='ij')
hoppings = J
hoppings = hoppings[np.newaxis,np.newaxis,:,:,:,:]
hoppings = np.repeat(hoppings,NUM_POINTS,0)
hoppings = np.repeat(hoppings,NUM_POINTS,1)

hamiltonian = np.zeros((NUM_POINTS,NUM_POINTS,3,3),dtype='complex')
n1 = np.linspace(-N,N,2*N+1,dtype='int')
n2 = np.linspace(-N,N,2*N+1,dtype='int')
kx = kx[:,:,np.newaxis,np.newaxis]
kx = np.repeat(kx,3,2)
kx = np.repeat(kx,3,3)
ky = ky[:,:,np.newaxis,np.newaxis]
ky = np.repeat(ky,3,2)
ky = np.repeat(ky,3,3)

for i in range(2*N+1):
    for j in range(2*N+1):
        exponent = np.exp(1j*(n1[i]*kx+n2[j]*ky))
        hamiltonian -= hoppings[:,:,i,j,:,:] * exponent

## Checking hermiticity
hermiticity_error = (np.sum(np.abs(hamiltonian 
                                  - np.conjugate(np.transpose(hamiltonian, 
                                                              (0,1,3,2))))) 
                                                              / NUM_POINTS**2)
print('Hermiticity error: ', hermiticity_error)

## Checcking reality
reality_error = np.sum(np.abs(np.imag(hamiltonian))) / NUM_POINTS**2
print('Reality error: ', reality_error)

# Diagonalising
print('Diagonalising...')
eigenvalues, blochvectors = np.linalg.eig(hamiltonian)
energies = np.real(eigenvalues)

# enforcing reality of the blochvectors
for k in range(3):
    for i in range(NUM_POINTS):
        for j in range(NUM_POINTS):
            phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,k], 
                                            blochvectors[i,j,:,k])))
            blochvectors[i,j,:,k] = np.real(blochvectors[i,j,:,k] * np.exp(-1j*phi))
blochvectors = np.real(blochvectors)

energies, blochvectors = sort_energy_grid(energies,blochvectors,regime='static')

# plotting
print('Plotting...')
plot_bandstructure2D(energies,a1,a2,'test.png',r=r,c=c,regime='static')

np.save('blochvectors.npy', blochvectors)

if False:
    blochvectors = gauge_fix_grid(blochvectors)
    k = np.linspace(0,1,NUM_POINTS,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    u = blochvectors[:,:,0,0]
    v = blochvectors[:,:,1,0]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()




