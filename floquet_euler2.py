import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg as la
from tight_binding.topology import dirac_string_rotation, energy_difference, sort_energy_grid, gauge_fix_grid
from tight_binding.bandstructure import plot_bandstructure2D

num_points = 100
num_steps = 100
lowest_quasi_energy = np.pi / 4
N = 2
T = 1
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
V1 = dirac_string_rotation(trivial_vectors, np.array([0.25, 0.25]), 
                           np.array([0.5,0]), 2, 0.5, num_points)
V1 = dirac_string_rotation(V1, np.array([0.25, 0.75]), np.array([0.5,0]), 2, 
                           0.5, num_points)

## creating a dirac string in between in anomalous gap
V2 = dirac_string_rotation(V1, np.array([0.5,0]), np.array([0,1]), 1, 0.25, 
                           num_points,True, np.array([[0.5,0.25],[0.5,0.75]]), 
                           np.array([[1,0],[1,0]]))

## creating a dirac string at the edge in the anomalous gap
V2 = dirac_string_rotation(V2, np.array([0,0]), np.array([0,1]), 1, 0.25, 
                           num_points)

# ENERGIES
trivial_energies = np.array([-2*np.pi/3, 0, 2*np.pi/3])
trivial_energies = trivial_energies[np.newaxis, np.newaxis, :]
trivial_energies = np.repeat(trivial_energies, num_points, 0)
trivial_energies = np.repeat(trivial_energies, num_points, 1)

differences = energy_difference(0.2, np.array([[0.25,0.25], [0.25, 0.75], 
                                             [0.75, 0.25], [0.75, 0.75]]), 
                                             2*np.pi/3, num_points)

E1 = np.zeros((num_points, num_points, 3), dtype='float')
E1[:,:,0] = trivial_energies[:,:,0] + differences
E1[:,:,1] = trivial_energies[:,:,1] - differences
E1[:,:,2] = trivial_energies[:,:,2]

E2 = np.zeros((num_points, num_points, 3), dtype='float')
E2[:,:,2] = 2*np.pi
E2[:,:,0] = -2*np.pi
E2[:,:,1] = E1[:,:,1] + E2[:,:,0] - E1[:,:,0]

# PLOTTING EVERYTHING FOR CHECKS
if True:
    plot_bandstructure2D(E1, a1, a2, 'test.png', regime='static',r=1,c=1)
    plot_bandstructure2D(E2, a1, a2, 'test.png', regime='static',r=1,c=1)
    k = np.linspace(0,1,num_points,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')

    u = V1[:,:,2,1]
    v = V1[:,:,1,1]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

    u = V2[:,:,0,2]
    v = V2[:,:,2,2]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

E1_diagonal = np.zeros((num_points, num_points, 3, 3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        E1_diagonal[i,j] = np.diag(E1[i,j])
E1 = E1_diagonal

E2_diagonal = np.zeros((num_points, num_points, 3, 3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        E2_diagonal[i,j] = np.diag(E2[i,j])
E2 = E2_diagonal


# FINDING THE HAMILTONIAN
H1A = np.matmul(V1, np.matmul(E1, np.transpose(V1, (0,1,3,2))))
H1B = np.matmul(V1, np.matmul(E2-E1, np.transpose(V1, (0,1,3,2))))
H2A = np.matmul(V2, np.matmul(2*E2-E1, np.transpose(V2, (0,1,3,2))))
H2B = np.matmul(V2, np.matmul(E1-E2, np.transpose(V2, (0,1,3,2))))

## Checking hermiticity
hermiticity_error = (np.sum(np.abs(H1A 
                                  - np.conjugate(np.transpose(H1A, 
                                                              (0,1,3,2))))) 
                                                              / num_points**2)
hermiticity_error += (np.sum(np.abs(H1B 
                                  - np.conjugate(np.transpose(H1B, 
                                                              (0,1,3,2))))) 
                                                              / num_points**2)
hermiticity_error += (np.sum(np.abs(H2A 
                                  - np.conjugate(np.transpose(H2A, 
                                                              (0,1,3,2))))) 
                                                              / num_points**2)
hermiticity_error += (np.sum(np.abs(H2B
                                  - np.conjugate(np.transpose(H2B, 
                                                              (0,1,3,2))))) 
                                                              / num_points**2)
print('Hermiticity error: ', hermiticity_error)

# FINDING THE HOPPINGS
k = np.linspace(-np.pi, np.pi, num_points)
kx, ky = np.meshgrid(k,k,indexing='ij')
dk = 2*np.pi / (num_points - 1)
J1A = np.zeros((2*N+1,2*N+1,3,3), dtype='complex')
J1B = np.zeros((2*N+1,2*N+1,3,3), dtype='complex')
J2A = np.zeros((2*N+1,2*N+1,3,3), dtype='complex')
J2B = np.zeros((2*N+1,2*N+1,3,3), dtype='complex')
for n1 in np.linspace(-N,N,2*N+1,endpoint=True, dtype='int'):
    for n2 in np.linspace(-N,N,2*N+1,endpoint=True, dtype='int'):
        exponent = -1j * (n1*kx + n2*ky)
        exponent = np.exp(exponent)
        exponent = exponent[:,:,np.newaxis, np.newaxis]
        exponent = np.repeat(exponent, 3, 2)
        exponent = np.repeat(exponent, 3, 3)
        integrand1A = H1A * exponent
        integrand1B = H1B * exponent
        integrand2A = H2A * exponent
        integrand2B = H2B * exponent
        J1A[n1 + N, n2 + N] =-(np.sum(integrand1A[:-1,:-1], (0,1))
                               *dk**2/(4*np.pi**2))
        J1B[n1 + N, n2 + N] =-(np.sum(integrand1B[:-1,:-1], (0,1))
                               *dk**2/(4*np.pi**2))
        J2A[n1 + N, n2 + N] =-(np.sum(integrand2A[:-1,:-1], (0,1))
                               *dk**2/(4*np.pi**2))
        J2B[n1 + N, n2 + N] =-(np.sum(integrand2B[:-1,:-1], (0,1))
                               *dk**2/(4*np.pi**2))
        
def J(t):
    if (t%1) < 0.5:
        hoppings = J1A + 4*t*J1B
    else:
        hoppings = J2A + 4*t*J2B
    return hoppings

# GOING BACKWARDS
# CALCULATING THE HAMILTONIAN
print('Calculating the hamiltonian backwards...')
hoppings = np.zeros((num_steps,2*N+1,2*N+1,3,3),dtype='complex')
for t in range(num_steps):
    hoppings[t] = J(t/num_steps*T)

hoppings = hoppings[:,np.newaxis,np.newaxis,:,:,:,:]
hoppings = np.repeat(hoppings,num_points,1)
hoppings = np.repeat(hoppings,num_points,2)

hamiltonian = np.zeros((num_steps,num_points,num_points,3,3),dtype='complex')
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
        exponent = exponent[np.newaxis,:,:,:,:]
        exponent = np.repeat(exponent,num_steps,0)
        hamiltonian -= hoppings[:,:,:,i,j,:,:] * exponent

## Checking hermiticity
hermiticity_error = (np.sum(np.abs(hamiltonian 
                                  - np.conjugate(np.transpose(hamiltonian, 
                                                              (0,1,2,4,3))))) 
                                                              / num_points**2 
                                                              / num_steps)
print('Hermiticity error: ', hermiticity_error)



# CALCULATING THE TIME EVOLUTION OPERATOR
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
error = np.sum(np.abs(identity - np.matmul(U,np.conjugate(np.transpose(U,(0,1,3,2)))))) / num_points**2
if error > 1e-5:
    print('High normalisation error!: {error}'.format(error=error))

# Checking reality of bloch vectors
error = np.sum(np.abs(np.transpose(U,(0,1,3,2)) - U)) / num_points**2
print('Reality error: {error}'.format(error=error))


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
plot_bandstructure2D(energies,a1,a2,'test.png',lowest_quasi_energy=lowest_quasi_energy,r=r,c=c)

np.save('blochvectors.npy', blochvectors)
np.save('J1A.npy', J1A)
np.save('J1B.npy', J1B)
np.save('J2A.npy', J2A)
np.save('J2B.npy', J2B)


# CALCULATING ZAK PHASES




