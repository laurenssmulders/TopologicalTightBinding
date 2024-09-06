import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from tight_binding.bandstructure import sort_energy_path
from tight_binding.utilities import compute_reciprocal_lattice_vectors_2D

J1A = np.load('J1A.npy')
J1B = np.load('J1B.npy')
J2A = np.load('J2A.npy')
J2B = np.load('J2B.npy')

def J(t):
    if (t%1) < 0.5:
        hoppings = J1A + 4*t*J1B
    else:
        hoppings = J2A + 4*t*J2B
    return hoppings

num_points = 100
num_steps = 100
L = 30
T = 1
N = int((J1A.shape[0] - 1) / 2)
n1 = np.linspace(-N,N,2*N+1,dtype='int')
n2 = np.linspace(-N,N,2*N+1,dtype='int')
lowest_quasi_energy = np.pi / 4
a1 = np.array([1,0])
a2 = np.array([0,1])



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
print(eigenvalues1.shape)
energies1 = np.real(1j*np.log(eigenvalues1))
energies1 = (energies1 + 2*np.pi*np.floor((lowest_quasi_energy-energies1) 
                                                / (2*np.pi) + 1))
energies2 = np.real(1j*np.log(eigenvalues2))
energies2 = (energies2 + 2*np.pi*np.floor((lowest_quasi_energy-energies2) 
                                                / (2*np.pi) + 1))

energies1, blochvectors1 = sort_energy_path(energies1,blochvectors1)
energies2, blochvectors2 = sort_energy_path(energies2,blochvectors2)

# Plotting
print('Plotting the finite structures...')
b1, b2 = compute_reciprocal_lattice_vectors_2D(a1, a2)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
k1 = np.linspace(0,np.linalg.norm(b1), 
                num_points+1)
k2 = np.linspace(0,np.linalg.norm(b2), 
                num_points+1)
energies1 = np.concatenate((energies1,np.array([energies1[0]])))
energies2 = np.concatenate((energies2,np.array([energies2[0]])))

top = lowest_quasi_energy + 2 * np.pi
bottom = lowest_quasi_energy
for band in range(3*L):
    distance_to_top1 = np.abs(energies1[:,band] - top)
    distance_to_bottom1 = np.abs(energies1[:,band] - bottom)
    
    threshold = 0.005 * 2 * np.pi
    discontinuity_mask1 = distance_to_top1 < threshold
    energies1[:,band] = np.where(discontinuity_mask1, np.nan, energies1[:,band])
    discontinuity_mask1 = distance_to_bottom1 < threshold
    energies1[:,band] = np.where(discontinuity_mask1, np.nan, energies1[:,band])

    distance_to_top2 = np.abs(energies2[:,band] - top)
    distance_to_bottom2 = np.abs(energies2[:,band] - bottom)
    
    threshold = 0.005 * 2 * np.pi
    discontinuity_mask2 = distance_to_top2 < threshold
    energies2[:,band] = np.where(discontinuity_mask2, np.nan, energies2[:,band])
    discontinuity_mask2 = distance_to_bottom2 < threshold
    energies2[:,band] = np.where(discontinuity_mask2, np.nan, energies2[:,band])

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
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
ax1.set_xlim(0,np.linalg.norm(b1))
ax2.set_xlim(0,np.linalg.norm(b2))
ax1.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
ax2.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
ax1.set_title('Cut along the x direction')
ax2.set_title('Cut along the y direction')
plt.show()
plt.close()


# Plotting the localisation
for state in range(energies1.shape[1]):
    loc = np.zeros(blochvectors1.shape[1]//3)
    #amplitudes = np.square(np.abs(blochvectors1[0,:,state])) #localisation at a specific k
    #for i in range(len(loc)):
        #   loc[i] = np.sum(amplitudes[3*i:3*i+3])
    for j in range(num_points):
        amplitudes = np.square(np.abs(blochvectors1[j,:,state]))
        kloc = np.zeros(blochvectors1.shape[1]//3)
        for i in range(len(kloc)):
            kloc[i] = np.sum(amplitudes[3*i:3*i+3])
        loc += kloc
    loc = loc / num_points # averaging localisation over all k

    #plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
    colours = list(np.zeros(energies1.shape[1]))
    for i in range(energies1.shape[1]):
        if i == state:
            ax1.plot(k1, energies1[:,i], c='magenta', zorder=10, linewidth=2)
        else:
            ax1.plot(k1, energies1[:,i], c='0', zorder=1)
    ax1.set_ylabel('$2\pi E / \omega$')
    ax1.set_yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
                3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                    '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
    ax1.set_xlabel('$k_x$')
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)

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
    for j in range(num_points):
        amplitudes = np.square(np.abs(blochvectors2[j,:,state]))
        kloc = np.zeros(blochvectors2.shape[1]//3)
        for i in range(len(kloc)):
            kloc[i] = np.sum(amplitudes[3*i:3*i+3])
        loc += kloc
    loc = loc / num_points # averaging localisation over all k

    #plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
    colours = list(np.zeros(energies2.shape[1]))
    for i in range(energies2.shape[1]):
        if i == state:
            ax1.plot(k2, energies2[:,i], c='magenta', zorder=10, linewidth=2)
        else:
            ax1.plot(k2, energies2[:,i], c='0', zorder=1)
    ax1.set_ylabel('$2\pi E / \omega$')
    ax1.set_yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
                3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                    '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
    ax1.set_xlabel('$k_y$')
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)

    positions = np.linspace(0,blochvectors2.shape[1] / 3,blochvectors2.shape[1] // 3)
    ax2.scatter(positions, loc)
    plt.title('Cut along the y direction')
    plt.savefig('edge_state_localisation_a2/edge_state_localisation_{state}'.format(state=state))
    plt.close(fig)