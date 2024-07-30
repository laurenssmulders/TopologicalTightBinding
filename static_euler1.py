import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import dirac_string_rotation, energy_difference, gauge_fix_grid
from tight_binding.bandstructure import sort_energy_grid
from tight_binding.utilities import rotate

num_points = 100
N = 1
r = 1
c = 1
divergence_threshold = 5
bands = [0,1]
kxmin = 0.1
kxmax = 0.9
kymin = 0.4
kymax = 0.6


# starting with trivial blochvectors
blochvectors = np.identity(3)
blochvectors = blochvectors[np.newaxis,np.newaxis,:,:]
blochvectors = np.repeat(blochvectors, num_points, 0)
blochvectors = np.repeat(blochvectors, num_points, 1)

# adding the dirac string at the edge
if True:
    rotation = np.zeros((num_points,num_points,3,3),dtype='float')
    for i in range(num_points):
        for j in range(num_points):
            rotation[i,j] = rotate(i / num_points * np.pi,blochvectors[i,j,:,0])
    blochvectors = np.matmul(rotation,blochvectors)

# adding two nodes
blochvectors = dirac_string_rotation(blochvectors,np.array([0.35,0.5]),np.array([0.3,0]),2,0.35,num_points)

#plotting the vectors
if False:
    k = np.linspace(0,1,num_points,endpoint=False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    u = blochvectors[:,:,1,0]
    v = blochvectors[:,:,2,0]
    plt.quiver(kx,ky,u,v, width=0.001)
    plt.show()

# finding the correct energies
energies = np.array([-1.,0.,1.])
energies = energies[np.newaxis,np.newaxis,:]
energies = np.repeat(energies, num_points, 0)
energies = np.repeat(energies, num_points, 1)

differences = energy_difference(0.1, np.array([[0.45,0.5],[0.55,0.5]]),1,num_points)
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


# calculating hamiltonian
diagonal_energies = np.zeros((num_points,num_points,3,3),dtype='float')
for i in range(num_points):
    for j in range(num_points):
        diagonal_energies[i,j] = np.diag(energies[i,j])

hamiltonian = np.matmul(blochvectors,np.matmul(diagonal_energies, np.transpose(blochvectors,(0,1,3,2))))

if True:
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

# Calculating patch euler class
if True:
    kx = np.linspace(kxmin,kxmax,num_points)
    dkx = kx[1] - kx[0]
    kx_extended = np.zeros(kx.shape[0] + 1)
    kx_extended[:-1] = kx
    kx_extended[-1] = kx[-1] + dkx
    kx = kx_extended

    ky = np.linspace(kymin,kymax,num_points)
    dky = ky[1] - ky[0]
    ky_extended = np.zeros(ky.shape[0] + 1)
    ky_extended[:-1] = ky
    ky_extended[-1] = ky[-1] + dky
    ky = ky_extended

    kx, ky = np.meshgrid(kx,ky,indexing='ij')


    # calculating the bandstructure on the patch
    hamiltonian = np.zeros((num_points+1,num_points+1,3,3),dtype='complex')
    for n1 in range(2*N+1):
        for n2 in range(2*N+1):
            exponent = np.exp(2*np.pi*1j*(n[n1]*kx + n[n2]*ky))
            exponent = exponent[:,:,np.newaxis,np.newaxis]
            exponent = np.repeat(exponent, 3, 2)
            exponent = np.repeat(exponent, 3, 3)
            hopping = J[n1,n2]
            hopping = hopping[np.newaxis,np.newaxis,:,:]
            hopping = np.repeat(hopping,num_points+1,0)
            hopping = np.repeat(hopping,num_points+1,1)
            hamiltonian = hamiltonian - hopping * exponent
    hamiltonian = np.real(hamiltonian)

    energies, blochvectors = np.linalg.eig(hamiltonian)
    energy_grid, blochvector_grid = sort_energy_grid(energies,blochvectors,'static')
    blochvector_grid = gauge_fix_grid(blochvector_grid)

    # Plotting energies
    E = energy_grid

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(kx, ky, E[...,0], cmap=cm.YlGnBu,
                                linewidth=0)
    surf2 = ax.plot_surface(kx, ky, E[...,1], cmap=cm.PuRd,
                                linewidth=0)
    surf3 = ax.plot_surface(kx, ky, E[...,2], cmap=cm.YlOrRd,
                                linewidth=0)
    tick_values = np.linspace(-4,4,9) * np.pi / 2
    tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    ax.set_zlim(np.nanmin(E),np.nanmax(E))
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()

    # calculating the x and y derivatives of the blochvectors (multiplied by dk)
    xder = np.zeros((num_points,num_points,3,3), dtype='float')
    for i in range(xder.shape[0]):
        for j in range(xder.shape[1]):
            xder[i,j] = (blochvector_grid[i+1,j] - blochvector_grid[i,j]) / dkx

    yder = np.zeros((num_points,num_points,3,3), dtype='float')
    for i in range(yder.shape[0]):
        for j in range(yder.shape[1]):
            yder[i,j] = (blochvector_grid[i,j+1] - blochvector_grid[i,j]) / dky

    # calculating Euler curvature at each point
    Eu = np.zeros((num_points,num_points), dtype='float')
    for i in range(Eu.shape[0]):
        for j in range(Eu.shape[1]):
            Eu[i,j] = (np.vdot(xder[i,j,:,bands[0]],yder[i,j,:,bands[1]])
                    - np.vdot(yder[i,j,:,bands[0]],xder[i,j,:,bands[1]]))

    # There will diverging derivative across Dirac strings: trying to remove them
    for i in range(Eu.shape[0]):
        for j in range(Eu.shape[1]):
            if np.abs(Eu[i,j]) > divergence_threshold:
                if j != 0:
                    Eu[i,j] = Eu[i,j-1]
                elif i != 0:
                    Eu[i,j] = Eu[i-1,j]
                else:
                    shift = 0
                    while np.abs(Eu[i,j]) > divergence_threshold and shift < 10:
                        shift += 1
                        Eu[i,j] = Eu[i + shift,j]
                if np.abs(Eu[i,j]) > divergence_threshold:
                        Eu[i,j] = 0

    # Doing y integrals
    integ_x = np.zeros(Eu.shape[0])
    for i in range(len(integ_x)):
        integ_x[i] = np.sum((Eu[i,1:] + Eu[i,:num_points-1]) / 2)*dky

    # Doing the x integral
    surface_term = 1 / (2*np.pi) * np.sum((integ_x[1:] 
                            + integ_x[:num_points-1]) / 2)*dkx

    # calculating the boundary term, dividing the boundary into 4 legs;
    # right, up, left, down
    right = np.zeros(num_points, dtype='float')
    up = np.zeros(num_points, dtype='float')
    left = np.zeros(num_points, dtype='float')
    down = np.zeros(num_points, dtype='float')

    for i in range(num_points):
        right[i] = np.vdot(blochvector_grid[i,0,:,bands[0]],xder[i,0,:,bands[1]])
        up[i] = np.vdot(blochvector_grid[-2,i,:,bands[0]],yder[-1,i,:,bands[1]])
        left[i] = - np.vdot(blochvector_grid[-2-i,-2,:,bands[0]],xder[-1-i,-1,:,bands[1]])
        down[i] = - np.vdot(blochvector_grid[0,-2-i,:,bands[0]],yder[0,-1-i,:,bands[1]])

    # Removing divergences here as well
    for i in range(len(right)):
        if np.abs(right[i]) > 5:
            if i != 0:
                right[i] = right[i-1]
            else:
                right[i] = right[i+5]
            if np.abs(right[i]) > 5:
                right[i] = 0
    for i in range(len(up)):
        if np.abs(up[i]) > 5:
            if i != 0:
                up[i] = up[i-1]
            else:
                up[i] = up[i+5]
            if np.abs(up[i]) > 5:
                up[i] = 0
    for i in range(len(left)):
        if np.abs(left[i]) > 5:
            if i != 0:
                left[i] = left[i-1]
            else:
                left[i] = left[i+5]
            if np.abs(left[i]) > 5:
                left[i] = 0
    for i in range(len(down)):
        if np.abs(down[i]) > 5:
            if i != 0:
                down[i] = down[i-1]
            else:
                down[i] = down[i+5]
            if np.abs(down[i]) > 5:
                down[i] = 0

    right = np.sum(right[1:] + right[:num_points-1]) / 2 * dkx
    up = np.sum(up[1:] + up[:num_points-1]) / 2 * dky
    left = np.sum(left[1:] + left[:num_points-1]) / 2 * dkx
    down = np.sum(down[1:] + down[:num_points-1]) / 2 * dky

    boundary_term = 1 / (2*np.pi) * (right + up + left + down)

    print('Surface Term:  ', surface_term)
    print('Boundary Term: ', boundary_term)
    chi = (surface_term - boundary_term)

    print('Euler Class: ', chi)

