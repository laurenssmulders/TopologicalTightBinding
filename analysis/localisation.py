import numpy as np
import matplotlib.pyplot as plt

energies = np.load('energies1.npy')
blochvectors = np.load('blochvectors1.npy')

edge_state_indices = [98,99]
NUM_POINTS = 100
L = energies.shape[1] // 3

k = np.linspace(-np.pi, np.pi, NUM_POINTS)

# FINDING THE LOCALISATION
positions = np.linspace(1,L,L)
print(positions.shape)
localisations = np.zeros((3*L,NUM_POINTS,
                            3,L),dtype='float') #k, sublattice, site
for state in edge_state_indices:
    for i in range(NUM_POINTS):
        for layer in range(L):
            localisations[state,i,0,layer] = np.abs(blochvectors[i,3*layer,state])**2
            localisations[state,i,1,layer] = np.abs(blochvectors[i,3*layer+1,state])**2
            localisations[state,i,2,layer] = np.abs(blochvectors[i,3*layer+2,state])**2

# AVERAGING OVER ALL K
if False:
    localisations = np.mean(localisations, axis=1)
    for state in edge_state_indices:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
        colours = list(np.zeros(energies.shape[1]))
        for i in range(energies.shape[1]):
            if i == state:
                ax1.plot(k, energies[:,i], c='magenta', zorder=10, linewidth=2)
            else:
                ax1.plot(k, energies[:,i], c='0', zorder=1)
        ax1.set_ylabel('$E$')
        ax1.set_xlabel('$k_x$')
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
        ax1.set_xlim(-np.pi, np.pi)

        ax2.scatter(positions, localisations[state,0], marker="+", label='A')
        ax2.scatter(positions, localisations[state,1], marker="+", label='B')
        ax2.scatter(positions, localisations[state,2], marker="+", label='C')
        ax2.set_ylabel('amplitude squared')
        ax2.set_xlabel('position')
        ax2.legend(loc='upper center')
        if True:
            plt.savefig('figures/{state}'.format(state=state))
        plt.show()
        plt.close(fig)

# AT K=0
if True:
    for state in edge_state_indices:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,15))
        colours = list(np.zeros(energies.shape[1]))
        for i in range(energies.shape[1]):
            if i == state:
                ax1.plot(k, energies[:,i], c='magenta', zorder=10, linewidth=2)
            else:
                ax1.plot(k, energies[:,i], c='0', zorder=1)
        ax1.set_ylabel('$E$')
        ax1.set_xlabel('$k_x$')
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
        ax1.set_xlim(-np.pi, np.pi)

        ax2.scatter(positions, localisations[state,25,0], marker="+", label='A')
        ax2.scatter(positions, localisations[state,25,1], marker="+", label='B')
        ax2.scatter(positions, localisations[state,25,2], marker="+", label='C')
        ax2.set_ylabel('amplitude squared')
        ax2.set_xlabel('position')
        ax2.legend(loc='upper center')
        if True:
            plt.savefig('figures/{state}'.format(state=state))
        plt.show()
        plt.close(fig)







