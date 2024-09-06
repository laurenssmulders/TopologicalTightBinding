import numpy as np
import matplotlib.pyplot as plt
from tight_binding.bandstructure import sort_energy_path

NUM_POINTS = 100
L = 30

J = np.load('J.npy')
N = (J.shape[0] - 1 ) // 2

# FINDING THE HOPPINGS OF THE RIBBON STRUCTURE
J_finite1 = np.zeros((2*N+1, 3*L, 3*L), dtype='complex')
for n1 in range(2*N+1):
    for i in range(3*L):
        for j in range(3*L):
            site_i = i // 3
            orbital_i = i % 3
            site_j = j //3
            orbital_j = j % 3
            n2 = site_j - site_i
            if abs(n2) > N:
                None
            else:
                n2 = n2 + N
                J_finite1[n1,i,j] = J[n1,n2,orbital_i,orbital_j]

J_finite2 = np.zeros((2*N+1, 3*L, 3*L), dtype='complex')
for n2 in range(2*N+1):
    for i in range(3*L):
        for j in range(3*L):
            site_i = i // 3
            orbital_i = i % 3
            site_j = j //3
            orbital_j = j % 3
            n1 = site_j - site_i
            if abs(n1) > N:
                None
            else:
                n1 = n1 + N
                J_finite2[n2,i,j] = J[n1,n2,orbital_i,orbital_j]

# FINDING THE HAMILTONIAN

n = np.linspace(-N,N,2*N+1)
k = np.linspace(-np.pi, np.pi, NUM_POINTS)

hamiltonian1 = np.zeros((NUM_POINTS, 3*L, 3*L), dtype='complex')
for n1 in range(2*N+1):
    for i in range(NUM_POINTS):
        hamiltonian1[i] -= J_finite1[n1] * np.exp(1j*n[n1]*k[i])

hamiltonian2 = np.zeros((NUM_POINTS, 3*L, 3*L), dtype='complex')
for n2 in range(2*N+1):
    for i in range(NUM_POINTS):
        hamiltonian2[i] -= J_finite2[n2] * np.exp(1j*n[n2]*k[i])

# CHECKING HERMITICITY AND REALITY

reality_error1 = (np.sum(np.abs(np.imag(hamiltonian1))) 
                  / np.sum(np.abs(hamiltonian1)))
reality_error2 = (np.sum(np.abs(np.imag(hamiltonian2))) 
                  / np.sum(np.abs(hamiltonian2)))

hermiticity_error1 = (np.sum(
    np.abs(hamiltonian1 - np.conjugate(np.transpose(hamiltonian1, (0,2,1))))) 
    / np.sum(np.abs(hamiltonian1)))
hermiticity_error2 = (np.sum(
    np.abs(hamiltonian2 - np.conjugate(np.transpose(hamiltonian2, (0,2,1))))) 
    / np.sum(np.abs(hamiltonian2)))

print('Hermiticity error 1: ', hermiticity_error1)
print('Hermiticity error 2: ', hermiticity_error2)
print('Reality error 1: ', reality_error1)
print('Reality error 2: ', reality_error2)

# DIAGONALISING
eigenvalues1, blochvectors1 = np.linalg.eig(hamiltonian1)
eigenvalues2, blochvectors2 = np.linalg.eig(hamiltonian2)

error1 = np.sum(np.abs(np.imag(eigenvalues1))) / np.sum(np.abs(eigenvalues1))
error2 = np.sum(np.abs(np.imag(eigenvalues2))) / np.sum(np.abs(eigenvalues2))
if error1 > 1e-10:
    print('Imaginary energies 1!    ', error1)
if error2 > 1e-10:
    print('Imaginary energies 2!    ', error2)

energies1 = np.real(eigenvalues1)
energies2 = np.real(eigenvalues2)

energies1, blochvectors1 = sort_energy_path(energies1, blochvectors1, 'static')
energies2, blochvectors2 = sort_energy_path(energies2, blochvectors2, 'static')

# PLOTTING
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))

ax1.plot(k,energies1,c='0')
ax2.plot(k,energies2,c='0')
ax1.set_ylabel('$E$')
ax2.set_ylabel('$E$')
ax1.set_xlabel = 'k_x'
ax2.set_xlabel = 'k_y'
ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
ax1.set_xlim(-np.pi,np.pi)
ax2.set_xlim(-np.pi,np.pi)
ax1.set_title('Cut along the x direction')
ax2.set_title('Cut along the y direction')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
plt.close()






