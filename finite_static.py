import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tight_binding.bandstructure import sort_energy_path
from tight_binding.utilities import compute_reciprocal_lattice_vectors_2D

J = np.load('J.npy')

NUM_POINTS = 100
L = 30
T = 1
N = int((J.shape[0] - 1) / 2)
n1 = np.linspace(-N,N,2*N+1,dtype='int')
n2 = np.linspace(-N,N,2*N+1,dtype='int')
lowest_quasi_energy = np.pi / 4
a1 = np.array([1,0])
a2 = np.array([0,1])



# Calculating the finite hamiltonions in both directions
print('Calculating the finite hamiltonians...')
# Along a1
hamiltonian1 = np.zeros((NUM_POINTS,3*L,3*L), dtype='complex')
for m2 in range(L):
    for i in range(NUM_POINTS):
        for k in range(2*N+1):
            for l in range(2*N+1):
                if m2+n2[l] >=0 and m2+n2[l] < L:
                    hamiltonian1[i,3*(m2+n2[l]):3*(m2+n2[l])+3,3*m2:3*m2+3] -= J[k,l]*np.exp(1j*2*np.pi*n1[k]*i/NUM_POINTS)

# Along a1
hamiltonian2 = np.zeros((NUM_POINTS,3*L,3*L), dtype='complex')
for m1 in range(L):
    for j in range(NUM_POINTS):
        for k in range(2*N+1):
            for l in range(2*N+1):
                if m1+n1[k] >=0 and m1+n1[k] < L:
                    hamiltonian2[j,3*(m1+n1[k]):3*(m1+n1[k])+3,3*m1:3*m1+3] -= J[k,l]*np.exp(1j*2*np.pi*n2[l]*j/NUM_POINTS)

## Checking hermiticity
hermiticity_error1 = (np.sum(np.abs(hamiltonian1 
                                  - np.conjugate(np.transpose(hamiltonian1, 
                                                              (0,2,1))))) 
                                                              / NUM_POINTS
                                                              / (3*L)**2)

hermiticity_error2 = (np.sum(np.abs(hamiltonian2 
                                  - np.conjugate(np.transpose(hamiltonian2, 
                                                              (0,2,1))))) 
                                                              / NUM_POINTS
                                                              / L**2)
print('Hermiticity errors: ', hermiticity_error1, hermiticity_error2)

## Checking reality
reality_error1 = np.sum(np.abs(np.imag(hamiltonian1))) / NUM_POINTS / L**2
reality_error2 = np.sum(np.abs(np.imag(hamiltonian2))) / NUM_POINTS / L**2
print('Reality errors: ', reality_error1, reality_error2)

## total sum for reference
reference1 = np.sum(np.abs(hamiltonian1)) / NUM_POINTS / L**2
reference2 = np.sum(np.abs(hamiltonian2)) / NUM_POINTS / L**2
print('References: ', reference1, reference2)

# diagonalising
print('Diagonalising the finite time evolution operators...')
eigenvalues1, blochvectors1 = np.linalg.eig(hamiltonian1)
eigenvalues2, blochvectors2 = np.linalg.eig(hamiltonian2)
energies1 = np.real(eigenvalues1)
energies2 = np.real(eigenvalues2)

energies1, blochvectors1 = sort_energy_path(energies1,blochvectors1,regime='static')
energies2, blochvectors2 = sort_energy_path(energies2,blochvectors2,regime='static')

# Plotting
print('Plotting the finite structures...')
b1, b2 = compute_reciprocal_lattice_vectors_2D(a1, a2)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
k1 = np.linspace(0,np.linalg.norm(b1), 
                NUM_POINTS+1)
k2 = np.linspace(0,np.linalg.norm(b2), 
                NUM_POINTS+1)
energies1 = np.concatenate((energies1,np.array([energies1[0]])))
energies2 = np.concatenate((energies2,np.array([energies2[0]])))

ax1.plot(k1,energies1,c='0')
ax2.plot(k2,energies2,c='0')
ax1.set_ylabel('$E$')
ax2.set_ylabel('$E$')
ax1.set_xlabel = 'k_x'
ax2.set_xlabel = 'k_y'
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
ax1.set_xlim(0,np.linalg.norm(b1))
ax2.set_xlim(0,np.linalg.norm(b2))
ax1.set_title('Cut along the x direction')
ax2.set_title('Cut along the y direction')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
plt.close()


# Plotting the localisation
for state in range(energies1.shape[1]):
    loc = np.zeros(blochvectors1.shape[1]//3)
    #amplitudes = np.square(np.abs(blochvectors1[0,:,state])) #localisation at a specific k
    #for i in range(len(loc)):
        #   loc[i] = np.sum(amplitudes[3*i:3*i+3])
    for j in range(NUM_POINTS):
        amplitudes = np.square(np.abs(blochvectors1[j,:,state]))
        kloc = np.zeros(blochvectors1.shape[1]//3)
        for i in range(len(kloc)):
            kloc[i] = np.sum(amplitudes[3*i:3*i+3])
        loc += kloc
    loc = loc / NUM_POINTS # averaging localisation over all k

    #plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
    colours = list(np.zeros(energies1.shape[1]))
    for i in range(energies1.shape[1]):
        if i == state:
            ax1.plot(k1, energies1[:,i], c='magenta', zorder=10, linewidth=2)
        else:
            ax1.plot(k1, energies1[:,i], c='0', zorder=1)
    ax1.set_ylabel('$E$')
    ax1.set_xlabel('$k_x$')
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
    ax1.set_xlim(0, 2*np.pi)

    positions = np.linspace(0,L-1,L)
    ax2.scatter(positions, loc)
    plt.title('Cut along the x direction')
    plt.savefig('edge_state_localisation_a1/edge_state_localisation_{state}'.format(state=state))
    plt.close(fig)

for state in range(energies2.shape[1]):
    loc = np.zeros(blochvectors2.shape[1]//3)
    #amplitudes = np.square(np.abs(blochvectors2[0,:,state])) #localisation at a specific k
    #for i in range(len(loc)):
        #   loc[i] = np.sum(amplitudes[3*i:3*i+3])
    for j in range(NUM_POINTS):
        amplitudes = np.square(np.abs(blochvectors2[j,:,state]))
        kloc = np.zeros(blochvectors2.shape[1]//3)
        for i in range(len(kloc)):
            kloc[i] = np.sum(amplitudes[3*i:3*i+3])
        loc += kloc
    loc = loc / NUM_POINTS # averaging localisation over all k

    #plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
    colours = list(np.zeros(energies2.shape[1]))
    for i in range(energies2.shape[1]):
        if i == state:
            ax1.plot(k2, energies2[:,i], c='magenta', zorder=10, linewidth=2)
        else:
            ax1.plot(k2, energies2[:,i], c='0', zorder=1)
    ax1.set_xlabel('$k_y$')
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
    ax1.set_xlim(0, 2*np.pi)

    positions = np.linspace(0,blochvectors2.shape[1] / 3,blochvectors2.shape[1] // 3)
    ax2.scatter(positions, loc)
    plt.title('Cut along the y direction')
    plt.savefig('edge_state_localisation_a2/edge_state_localisation_{state}'.format(state=state))
    plt.close(fig)