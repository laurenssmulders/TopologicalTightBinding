import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import dirac_string_rotation
from tight_binding.topology import energy_difference, gauge_fix_grid
from tight_binding.bandstructure import plot_bandstructure2D, sort_energy_grid

num_points = 100
N = 2
a1 = np.array([1,0])
a2 = np.array([0,1])
bands = [0,1]
divergence_threshold = 10

# FINDING THE BLOCHVECTORS
blochvectors = np.identity(3)
blochvectors = blochvectors[np.newaxis,np.newaxis,:,:]
blochvectors = np.repeat(blochvectors, num_points, 0)
blochvectors = np.repeat(blochvectors, num_points, 1)

if False:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.5,0]), np.array([0,1]), 0, 0.5, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25, 0.25]), 
                                            np.array([0.5, 0]), 2, 0.3, num_points, 
                                            True, np.array([[0.5,0.25]]), 
                                            np.array([[0,1]]))
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25, 0.75]), 
                                            np.array([0.5, 0]), 2, 0.3, num_points, 
                                            True, np.array([[0.5,0.75]]), 
                                            np.array([[0,1]]))
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.5,1]), 
                                            np.array([0,-1]), 0, 0.2, num_points, 
                                            True, 
                                            np.array([[0.5, 0.25],[0.5, 0.75]]), 
                                            np.array([[1,0], [1,0]]))

if True:
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25, 0.25]), 
                                            np.array([0.5, 0]), 2, 0.5, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.25, 0.75]), 
                                            np.array([0.5, 0]), 2, 0.5, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0,0]), np.array([0,1]), 0, 0.25, num_points)
    blochvectors = dirac_string_rotation(blochvectors, np.array([0.5,1]), 
                                            np.array([0,-1]), 0, 0.25, num_points, 
                                            True, 
                                            np.array([[0.5, 0.25],[0.5, 0.75]]), 
                                            np.array([[1,0], [1,0]]))

if True:
    k = np.linspace(0,1,num_points,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    u = blochvectors[:,:,2,1]
    v = blochvectors[:,:,1,1]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

# FINDING THE ENERGIES
energies = np.array([-1.,0.,1.])
energies = energies[np.newaxis,np.newaxis,:]
energies = np.repeat(energies, num_points, 0)
energies = np.repeat(energies, num_points, 1)

differences = energy_difference(0.2, np.array([[0.25,0.25], [0.25, 0.75], 
                                             [0.75, 0.25], [0.75, 0.75]]), 
                                             1, num_points)
energies[:,:,0] = energies[:,:,0] + differences
energies[:,:,1] = energies[:,:,1] - differences

plot_bandstructure2D(energies, a1, a2, 'test.png', regime='static',r=1,c=1)

# FINDING THE HAMILTONIAN
diagonal_energies = np.zeros((num_points, num_points, 3, 3), dtype='float')
for i in range(num_points):
    for j in range(num_points):
        diagonal_energies[i,j] = np.diag(energies[i,j])

hamiltonian = np.matmul(blochvectors, 
                        np.matmul(diagonal_energies, 
                                  np.transpose(blochvectors, (0,1,3,2))))

# FINDING THE TUNNELINGS
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
        integrand = hamiltonian * exponent
        J[n1 + N, n2 + N] =-np.sum(integrand[:-1,:-1], (0,1))*dk**2/(4*np.pi**2)

# FINDING THE TRUNCATED HAMILTONIAN
hamiltonian = np.zeros((num_points, num_points, 3, 3), dtype='complex')
for n1 in np.linspace(-N,N,2*N+1,endpoint=True, dtype='int'):
    for n2 in np.linspace(-N,N,2*N+1,endpoint=True, dtype='int'):
        exponent = 1j * (n1*kx + n2*ky)
        exponent = np.exp(exponent)
        exponent = exponent[:,:,np.newaxis, np.newaxis]
        exponent = np.repeat(exponent, 3, 2)
        exponent = np.repeat(exponent, 3, 3)
        hamiltonian -= J[n1 + N, n2 + N] * exponent

# CHECKING REALITY AND HERMITICITY
reality_error = (np.sum(np.abs(hamiltonian - np.conjugate(hamiltonian))) 
                 / num_points**2)
hermiticity_error = (np.sum(np.abs(hamiltonian 
                                   - np.conjugate(np.transpose(hamiltonian, 
                                                               (0,1,3,2)))))
                                                               / num_points**2)
print('Reality error: ', reality_error)
print('Hermiticity error: ', hermiticity_error)

# CALCULATING THE BANDSTRUCTURE
energies, blochvectors = np.linalg.eig(hamiltonian)
energies = np.real(energies)

# MAKING THE BLOCHVECTORS REAL
for i in range(num_points):
    for j in range(num_points):
        for l in range(3):
            blochvectors[i,j,:,l] = np.conjugate(np.sqrt(np.dot(
                blochvectors[i,j,:,l], 
                blochvectors[i,j,:,l])))*blochvectors[i,j,:,l]
            
# CHECKING BLOCHVECTOR REALITY AND ORTHOGONALITY
identity = np.identity(3)
identity = identity[np.newaxis, np.newaxis, :, :]
identity = np.repeat(identity, num_points, 0)
identity = np.repeat(identity, num_points, 1)

reality_error = (np.sum(np.abs(blochvectors - np.conjugate(blochvectors))) 
                 / num_points**2)
orthogonality_error = (np.sum(np.abs(np.matmul(np.transpose(blochvectors, 
                                                            (0,1,3,2)), 
                                                            blochvectors) 
                                                            - identity)) 
                                                            / num_points**2)
print('Reality error: ', reality_error)
print('Orthogonality error: ', orthogonality_error)

energies, blochvectors = sort_energy_grid(energies, blochvectors, 'static')

# PLOTTING THE ENERGIES
plot_bandstructure2D(energies,a1,a2,'test.png',regime='static', r=5, c=5)

# CALCULATING EULER CLASS
blochvectors = gauge_fix_grid(blochvectors)
dk = 2*np.pi / (num_points - 1)
xder = (blochvectors[1:] - blochvectors[:-1]) / dk
yder = (blochvectors[:,1:] - blochvectors[:,:-1]) / dk
Eu = np.zeros((num_points - 1, num_points - 1), dtype='float')
for i in range(num_points - 1):
    for j in range(num_points - 1):
        Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]], yder[i,j,:,bands[1]])
                   - np.vdot(yder[i,j,:,bands[0]], xder[i,j,:,bands[1]]))
        if np.abs(Eu[i,j]) > divergence_threshold:
            Eu[i,j] = 0

# Plotting Euler curvature
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x,y = np.meshgrid(range(num_points-1),range(num_points-1),indexing='ij')
surf1 = ax.plot_surface(x,y, Eu, cmap=cm.YlGnBu,
                            linewidth=0)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.grid(False)
ax.set_box_aspect([1, 1, 2])
plt.show()
plt.close()

Eu = Eu * dk**2
chi = np.sum(Eu) / (2*np.pi)
print('Euler Class: ', chi)