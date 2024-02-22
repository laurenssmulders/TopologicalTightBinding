import numpy as np
import os
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven_finite_x
from tight_binding.bandstructure import sort_energy_path
from tight_binding.diagonalise import compute_eigenstates


# PARAMETERS
L = 30
delta_A = -7
delta_C = 1
omega = 10
A_x = 1

delta_B = -delta_A - delta_C

## OTHER
a_1 = np.array([1,0])
a_2 = np.array([0,1])
num_points = 101
num_steps = 100
num_lines = 100
lowest_quasi_energy = - np.pi
offsets = np.zeros((3,2))

# BLOCH HAMILTONIAN
H = square_hamiltonian_driven_finite_x(
    L=L,
    delta_a=delta_A,
    delta_b=delta_B,
    delta_c=delta_C,
    J_ab_0=1,
    J_ac_0=1,
    J_bc_0=1,
    J_ac_1x=1,
    J_bc_1y=1,
    J_ab_2m=1,
    A_x=A_x,
    omega=omega
)

k = np.linspace(-np.pi, np.pi, num_points)
E = np.zeros((num_points, 3*L))
blochvectors = np.zeros((num_points, 3*L, 3*L), dtype='complex')

for i in range(len(k)):
    energies, eigenvectors = compute_eigenstates(H,k[i],omega,num_steps,
                                                 lowest_quasi_energy,False,method='Runge-Kutta')
    E[i] = energies
    blochvectors[i] = eigenvectors

E, blochvectors = sort_energy_path(E,blochvectors)

plt.plot(k,E,c='0')
plt.ylabel('$2\pi E / \omega$')
plt.yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
            3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                  '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
plt.xlabel('$k_y$')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
plt.xlim(-np.pi, np.pi)
plt.ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
plt.show()

# Average localisation of the two edge states at k=0
ind = np.argsort(E[50])
psi1 = blochvectors[:,:,ind][50,:,2*L]
psi2 = blochvectors[:,:,ind][50,:,2*L+1]

localisation = np.zeros(L)
positions = np.linspace(0,L-1,L)
for i in range(L):
    localisation[i] = 1/3*((np.abs(psi1[3*i] + psi1[3*i+1] + psi1[3*i+2]))**2
                           +(np.abs(psi2[3*i] + psi2[3*i+1] + psi2[3*i+2]))**2)

plt.close()
plt.scatter(positions, localisation)
plt.show()

