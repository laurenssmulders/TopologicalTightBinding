import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import dirac_string_rotation, energy_difference
from tight_binding.bandstructure import sort_energy_grid

num_points = 100
N = 1
r = 1
c = 1

# starting with trivial blochvectors
blochvectors = np.identity(3)
blochvectors = blochvectors[np.newaxis,np.newaxis,:,:]
blochvectors = np.repeat(blochvectors, num_points, 0)
blochvectors = np.repeat(blochvectors, num_points, 1)

node_neg = np.array([0.25,0.5])
ds = np.array([0.5,0])
blochvectors = dirac_string_rotation(blochvectors,node_neg,ds,2,0.5,num_points)

if False:
    #plotting the vectors
    u = blochvectors[:,:,0,0]
    v = blochvectors[:,:,1,0]
    k = np.linspace(0,1,num_points)
    kx,ky = np.meshgrid(k,k,indexing='ij')
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

# finding the correct energies
energies = np.array([-1.,0.,1.])
energies = energies[np.newaxis,np.newaxis,:]
energies = np.repeat(energies, num_points, 0)
energies = np.repeat(energies, num_points, 1)

differences = energy_difference(0.1, np.array([[0.25,0.5],[0.75,0.5]]),1,num_points)
energies[:,:,0] += differences
energies[:,:,1] -= differences

if False:
    # plotting energies
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    k = np.linspace(0,1,num_points)
    kx,ky = np.meshgrid(k,k,indexing='ij')
    surf1 = ax.plot_surface(kx, ky, energies[:,:,0], cmap=cm.YlOrRd, edgecolor='darkred',
                            linewidth=0, rstride=1, cstride=1)
    surf2 = ax.plot_surface(kx, ky, energies[:,:,1], cmap=cm.PuRd, edgecolor='purple',
                            linewidth=0, rstride=1, cstride=1)
    surf3 = ax.plot_surface(kx, ky, energies[:,:,2], cmap=cm.YlGnBu, edgecolor='darkblue',
                                    linewidth=0, rstride=1, cstride=1)

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5})

    plt.show()
    plt.close()

diagonal_energies = np.zeros((num_points,num_points,3,3),dtype='float')
for i in range(num_points):
    for j in range(num_points):
        diagonal_energies[i,j] = np.diag(energies[i,j])

hamiltonian = np.matmul(blochvectors,np.matmul(diagonal_energies, np.transpose(blochvectors,(0,1,3,2))))

J = np.zeros((2*N+1,2*N+1,3,3),dtype='complex')
n = np.linspace(-N,N,2*N+1,dtype='int')
for n1 in range(2*N+1):
    for n2 in range(2*N+1):
        k = np.linspace(0,1,num_points-1,False)
        kx,ky = np.meshgrid(k,k,indexing='ij')
        exponent = np.exp(-2*np.pi*1j*(n[n1]*kx + n[n2]*ky))
        exponent = exponent[:,:,np.newaxis,np.newaxis]
        exponent = np.repeat(exponent, 3, 2)
        exponent = np.repeat(exponent, 3, 3)
        integrand = hamiltonian[:-1,:-1,:,:] * exponent
        dk = 1 / num_points
        J[n1,n2] = - np.sum(integrand,(0,1)) *dk**2

# calculating hamiltonian backwards
hamiltonian = np.zeros((num_points,num_points,3,3),dtype='complex')
for n1 in range(2*N+1):
    for n2 in range(2*N+1):
        k = np.linspace(0,1,num_points)
        kx,ky = np.meshgrid(k,k,indexing='ij')
        exponent = np.exp(2*np.pi*1j*(n[n1]*kx + n[n2]*ky))
        exponent = exponent[:,:,np.newaxis,np.newaxis]
        exponent = np.repeat(exponent, 3, 2)
        exponent = np.repeat(exponent, 3, 3)
        hopping = J[n1,n2]
        hopping = hopping[np.newaxis,np.newaxis,:,:]
        hopping = np.repeat(hopping,num_points,0)
        hopping = np.repeat(hopping,num_points,1)
        hamiltonian = hamiltonian - hopping * exponent
hamiltonian = np.real(hamiltonian)

energies, blochvectors = np.linalg.eig(hamiltonian)
energies, blochvectors = sort_energy_grid(energies,blochvectors,'static')

if True:
    # plotting energies
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    k = np.linspace(0,1,num_points)
    kx,ky = np.meshgrid(k,k,indexing='ij')
    surf1 = ax.plot_surface(kx, ky, energies[:,:,0], cmap=cm.YlOrRd, edgecolor='darkred',
                            linewidth=0, rstride=r, cstride=c)
    surf2 = ax.plot_surface(kx, ky, energies[:,:,1], cmap=cm.PuRd, edgecolor='purple',
                            linewidth=0, rstride=r, cstride=c)
    surf3 = ax.plot_surface(kx, ky, energies[:,:,2], cmap=cm.YlGnBu, edgecolor='darkblue',
                                    linewidth=0, rstride=r, cstride=c)

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5})

    plt.show()
    plt.close()

if True:
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




        