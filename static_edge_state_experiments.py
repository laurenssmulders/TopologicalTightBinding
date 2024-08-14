import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg as la
import copy
from tight_binding.topology import dirac_string_rotation, energy_difference, sort_energy_grid, gauge_fix_grid
from tight_binding.bandstructure import plot_bandstructure2D

num_points = 100
N = 1
a1 = np.array([1,0])
a2 = np.array([0,1])
r=1
c=1

# 01 FINDING ENERGIES AND BLOCHVECTORS FOR EVERY STEP
# VECTORS
## starting with trivial blochvectors
trivial_vectors = np.identity(3)
trivial_vectors = trivial_vectors[np.newaxis, np.newaxis, :, :]
trivial_vectors = np.repeat(trivial_vectors, num_points, 0)
trivial_vectors = np.repeat(trivial_vectors, num_points, 1)

## creating two pairs of nodes in gap 1
V = trivial_vectors
#V = dirac_string_rotation(trivial_vectors, np.array([0.25, 0.5]), 
#                           np.array([0.5,0]), 2, 0.5, num_points)

# ENERGIES
trivial_energies = np.array([-2*np.pi/3, 0, 2*np.pi/3])
trivial_energies = trivial_energies[np.newaxis, np.newaxis, :]
trivial_energies = np.repeat(trivial_energies, num_points, 0)
trivial_energies = np.repeat(trivial_energies, num_points, 1)

differences = energy_difference(0.2, np.array([[0.25,0.5], [0.75, 0.5]]), 
                                             2*np.pi/3, num_points)

E = np.zeros((num_points, num_points, 3), dtype='float')
E[:,:,0] = trivial_energies[:,:,0]# + differences
E[:,:,1] = trivial_energies[:,:,1]# - differences
E[:,:,2] = trivial_energies[:,:,2]

# PLOTTING EVERYTHING FOR CHECKS
if True:
    plot_bandstructure2D(E, a1, a2, 'test.png', regime='static',r=1,c=1)
    k = np.linspace(0,1,num_points,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')

    #u = V[:,:,0,0]
    #v = V[:,:,1,0]
    #plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

E_diagonal = np.zeros((num_points, num_points, 3, 3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        E_diagonal[i,j] = np.diag(E[i,j])
E = E_diagonal


# FINDING THE HAMILTONIAN
H = np.matmul(V, np.matmul(E, np.transpose(V, (0,1,3,2))))

## Checking hermiticity
hermiticity_error = (np.sum(np.abs(H 
                                  - np.conjugate(np.transpose(H, 
                                                              (0,1,3,2))))) 
                                                              / num_points**2)
print('Hermiticity error: ', hermiticity_error)

# FINDING THE HOPPINGS
k = np.linspace(-np.pi, np.pi, num_points)
kx, ky = np.meshgrid(k,k,indexing='ij')
dk = 2*np.pi / (num_points - 1)
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
        
print(J)

# GOING BACKWARDS
# CALCULATING THE HAMILTONIAN
print('Calculating the hamiltonian backwards...')
hoppings = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
hoppings = J
hoppings = hoppings[np.newaxis,np.newaxis,:,:,:,:]
hoppings = np.repeat(hoppings,num_points,1)
hoppings = np.repeat(hoppings,num_points,2)

hamiltonian = np.zeros((num_points,num_points,3,3),dtype='complex')
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

print(hamiltonian)
## Checking hermiticity
hermiticity_error = (np.sum(np.abs(hamiltonian 
                                  - np.conjugate(np.transpose(hamiltonian, 
                                                              (0,1,3,2))))) 
                                                              / num_points**2)
print('Hermiticity error: ', hermiticity_error)

# Diagonalising
print('Diagonalising...')
eigenvalues, blochvectors = np.linalg.eig(hamiltonian)
energies = np.real(eigenvalues)

# enforcing reality of the blochvectors
for k in range(3):
    for i in range(num_points):
        for j in range(num_points):
            phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,k], 
                                            blochvectors[i,j,:,k])))
            blochvectors[i,j,:,k] = np.real(blochvectors[i,j,:,k] * np.exp(-1j*phi))
blochvectors = np.real(blochvectors)

energies, blochvectors = sort_energy_grid(energies,blochvectors,regime='static')

# plotting
print('Plotting...')
plot_bandstructure2D(energies,a1,a2,'test.png',r=r,c=c,regime='static')

np.save('blochvectors.npy', blochvectors)
np.save('J.npy', J)




