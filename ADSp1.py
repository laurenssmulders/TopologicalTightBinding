'''This script is to generate the figures relating to the ADS+1 phase I found
through trial and error.

Inspired by Kagome, I had defined J the tunneling between all three ABC 
sublattices at the same site. DeltaA, DeltaB, DeltaC the onsite potentials.
A tunneling J1x between A and C in the x direction and J1y between B and C in 
the y direction. And finally a tunneling J2 between A and B in the x + y 
direction. The Bloch Hamiltonian is then:

    a1 = (1,0)T, a2 = (0,1)T

   | DeltaA                  -2(J+J2cos(k.(a1+a2)))        -2(J+J1xcos(k.a1))  | 
   |                                                                           |
   | -2(J+J2cos(k.(a1+a2)))          DeltaB                 -2(J+J1ycos(k.a2)) |
   |                                                                           |
   | -2(J+J1xcos(k.a1))          -2(J+J1ycos(k.a2))                DeltaC      |

Then I defined a drive:
A(t) = (Ax,-Ay)cos(wt)
giving k --> k+A(t)

Defined parameters:
J = 1
J1x = 1 + dJ1x
J1y = 1 + dJ1y
J2 = 1 + dJ2
DeltaB = -DeltaA - DeltaC
Ax = Ay = 1

Then varied parameters dJ1x, dJ1y, dJ2, DeltaA, DeltaC and w.

The final ADSp1 phase is at w = 9, dJ1y=-dJ1x=0.7, dJ2 = -0.9, DeltaA=-DeltaC=3

Want to deform from w = 20, and the rest 0, analysing Euler classes etc. at 
every step.
'''

# IMPORTS
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm

from tight_binding.bandstructure import sort_energy_grid, plot_bandstructure2D, locate_nodes, sort_energy_path
from tight_binding.utilities import compute_reciprocal_lattice_vectors_2D
from tight_binding.topology import gauge_fix_grid


# PARAMETERS
plot = False
loc_nodes = False
euler = False
zak = False
static = False
finite = True

num_points = 100
num_steps = 100
lowest_quasi_energy = -3*np.pi/8
r = 1
c = 1
node_threshold = 0.05

omega = 9
dJ1x = -0.7
dJ1y = 0.7
dJ2 = -0.9
dA = 3
dC = -3

dB = -dA-dC
J1x = 1+dJ1x
J1y = 1+dJ1y
J2 = 1+dJ2
T = 2*np.pi/omega
a1 = np.array([1,0])
a2 = np.array([0,1])

# euler
kxmin = -np.pi
kxmax = np.pi
kymin = 0.5*np.pi
kymax = np.pi
bands = [1,2]
divergence_threshold = 5

# finite
L = 101

if plot:
    # HAMILTONIAN
    print('Calculating hamiltonian...')
    def A(t):
        return np.array([1,-1])*np.cos(omega*t)

    hamiltonian = np.zeros((num_steps,num_points,num_points,3,3),dtype='float')
    k = np.linspace(0,2*np.pi,num_points,False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    t = np.linspace(0,T,num_steps,False)

    for s in range(num_steps):
        for i in range(num_points):
            for j in range(num_points):
                k = np.array([kx[i,j],ky[i,j]])
                hamiltonian[s,i,j] = np.array(
                    [
                        [dA, -2*(1+J2*np.cos(np.vdot(k+A(t[s]),a1+a2))), -2*(1+J1x*np.cos(np.vdot(k+A(t[s]),a1)))],
                        [-2*(1+J2*np.cos(np.vdot(k+A(t[s]),a1+a2))), dB, -2*(1+J1y*np.cos(np.vdot(k+A(t[s]),a2)))],
                        [-2*(1+J1x*np.cos(np.vdot(k+A(t[s]),a1))), -2*(1+J1y*np.cos(np.vdot(k+A(t[s]),a2))), dC]
                    ]
                )

    # TIME EVOLUTION
    print('Calculating time evolution...')
    dt = T / num_steps
    U = np.identity(3, dtype='complex')
    U = U[np.newaxis,np.newaxis,:,:]
    U = np.repeat(U,num_points,0)
    U = np.repeat(U,num_points,1)

    for s in range(num_steps):
        U = np.matmul(la.expm(-1j*hamiltonian[s]*dt), U)

    # DIAGONALISING
    print('Diagonalising...')
    eigenvalues, eigenvectors = np.linalg.eig(U)
    energies = np.real(1j*np.log(eigenvalues))
    error = np.sum(np.abs(np.real(np.log(eigenvalues)))) / num_points**2
    if error > 1e-5:
        print('Imaginary quasi-energies!    {error}'.format(error=error))
    energies = (energies + 2*np.pi*np.floor((lowest_quasi_energy-energies) 
                                                    / (2*np.pi) + 1))

    blochvectors = eigenvectors
    for i in range(num_points):
        for j in range(num_points):
            for band in range(3):
                phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,band], 
                                                blochvectors[i,j,:,band])))
                blochvectors[i,j,:,band] = np.real(blochvectors[i,j,:,band] * np.exp(-1j*phi))
    blochvectors = np.real(blochvectors)

    energies, blochvectors = sort_energy_grid(energies, blochvectors)

    plot_bandstructure2D(energies, a1, a2, 'test.png', lowest_quasi_energy=lowest_quasi_energy,discontinuity_threshold=0.01,r=r,c=c)

if euler:
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

    # HAMILTONIAN
    print('Calculating hamiltonian...')
    def A(t):
        return np.array([1,-1])*np.cos(omega*t)

    hamiltonian = np.zeros((num_steps,num_points+1,num_points+1,3,3),dtype='float')
    t = np.linspace(0,T,num_steps,False)

    for s in range(num_steps):
        for i in range(num_points+1):
            for j in range(num_points+1):
                k = np.array([kx[i,j],ky[i,j]])
                hamiltonian[s,i,j] = np.array(
                    [
                        [dA, -2*(1+J2*np.cos(np.vdot(k+A(t[s]),a1+a2))), -2*(1+J1x*np.cos(np.vdot(k+A(t[s]),a1)))],
                        [-2*(1+J2*np.cos(np.vdot(k+A(t[s]),a1+a2))), dB, -2*(1+J1y*np.cos(np.vdot(k+A(t[s]),a2)))],
                        [-2*(1+J1x*np.cos(np.vdot(k+A(t[s]),a1))), -2*(1+J1y*np.cos(np.vdot(k+A(t[s]),a2))), dC]
                    ]
                )

    # TIME EVOLUTION
    print('Calculating time evolution...')
    dt = T / num_steps
    U = np.identity(3, dtype='complex')
    U = U[np.newaxis,np.newaxis,:,:]
    U = np.repeat(U,num_points+1,0)
    U = np.repeat(U,num_points+1,1)

    for s in range(num_steps):
        U = np.matmul(la.expm(-1j*hamiltonian[s]*dt), U)

    # DIAGONALISING
    print('Diagonalising...')
    eigenvalues, eigenvectors = np.linalg.eig(U)
    energies = np.real(1j*np.log(eigenvalues))
    error = np.sum(np.abs(np.real(np.log(eigenvalues)))) / num_points**2
    if error > 1e-5:
        print('Imaginary quasi-energies!    {error}'.format(error=error))
    energies = (energies + 2*np.pi*np.floor((lowest_quasi_energy-energies) 
                                                    / (2*np.pi) + 1))

    blochvectors = eigenvectors
    for i in range(num_points+1):
        for j in range(num_points+1):
            for band in range(3):
                phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,band], 
                                                blochvectors[i,j,:,band])))
                blochvectors[i,j,:,band] = np.real(blochvectors[i,j,:,band] * np.exp(-1j*phi))
    blochvectors = np.real(blochvectors)

    energies, blochvectors = sort_energy_grid(energies, blochvectors)
    
    # Plotting energies
    E = energies
    top = lowest_quasi_energy + 2 * np.pi
    bottom = lowest_quasi_energy
    for band in range(E.shape[0]):
        distance_to_top = np.abs(E[band] - top)
        distance_to_bottom = np.abs(E[band] - bottom)
        
        threshold = 0.05 * 2 * np.pi
        discontinuity_mask = distance_to_top < threshold
        E[band] = np.where(discontinuity_mask, np.nan, E[band])
        discontinuity_mask = distance_to_bottom < threshold
        E[band] = np.where(discontinuity_mask, np.nan, E[band])

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
    ztick_labels = ['$-2\pi$', '$-3\pi/2$', '$-\pi$', '$-\pi/2$', '0', 
                    '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
    ax.set_zticks(tick_values)
    ax.set_zticklabels(ztick_labels)
    ax.set_zlim(np.nanmin(E),np.nanmax(E))
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()
    
    # gauge fixing
    blochvector_grid = gauge_fix_grid(blochvectors)
    
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

    # Plotting Euler Curvature
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(kx[:-1,:-1], ky[:-1,:-1], Eu, cmap=cm.YlGnBu,
                                linewidth=0)
    tick_values = np.linspace(-4,4,9) * np.pi / 2
    tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    plt.show()
    plt.close()

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
    print('Patch Euler Class: ', chi)

if loc_nodes:
    locate_nodes(energies, a1, a2, 'test.png', node_threshold)

if zak:
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


if static:
    # HAMILTONIAN
    print('Calculating hamiltonian...')
    hamiltonian = np.zeros((num_points,num_points,3,3),dtype='float')
    k = np.linspace(0,2*np.pi,num_points,False)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    for i in range(num_points):
        for j in range(num_points):
            k = np.array([kx[i,j],ky[i,j]])
            hamiltonian[i,j] = np.array(
                [
                    [dA, -2*(1+J2*np.cos(np.vdot(k,a1+a2))), -2*(1+J1x*np.cos(np.vdot(k,a1)))],
                    [-2*(1+J2*np.cos(np.vdot(k,a1+a2))), dB, -2*(1+J1y*np.cos(np.vdot(k,a2)))],
                    [-2*(1+J1x*np.cos(np.vdot(k,a1))), -2*(1+J1y*np.cos(np.vdot(k,a2))), dC]
                ]
            )

    # DIAGONALISING
    print('Diagonalising...')
    energies, blochvectors = np.linalg.eig(hamiltonian)
    for i in range(num_points):
        for j in range(num_points):
            for band in range(3):
                phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,band], 
                                                blochvectors[i,j,:,band])))
                blochvectors[i,j,:,band] = np.real(blochvectors[i,j,:,band] * np.exp(-1j*phi))
    blochvectors = np.real(blochvectors)

    energies, blochvectors = sort_energy_grid(energies, blochvectors)

    plot_bandstructure2D(energies, a1, a2, 'test.png', regime='static',r=r,c=c)



# FINITE STRUCTURES
if finite:
    J_0_0 = np.array([
        [-dA,2,2],
        [2,-dB,2],
        [2,2,-dC]
    ])

    J_1_1 = np.array([
        [0,J2,0],
        [J2,0,0],
        [0,0,0]
    ])

    J_1_0 = np.array([
        [0,0,J1x],
        [0,0,0],
        [J1x,0,0]
    ])

    J_0_1 = np.array([
        [0,0,0],
        [0,0,J1y],
        [0,J1y,0]
    ])

    def A(time):
        return np.array([1,-1])*np.cos(omega*time)
    
    def J(time):
        hoppings = np.zeros((3,3,3,3),dtype='complex')
        hoppings[0,0] = J_1_1 * np.exp(1j*np.vdot(A(time),-a1-a2))
        hoppings[2,2] = J_1_1 * np.exp(1j*np.vdot(A(time),a1+a2))
        hoppings[2,1] = J_1_0 * np.exp(1j*np.vdot(A(time),a1))
        hoppings[0,1] = J_1_0 * np.exp(1j*np.vdot(A(time),-a1))
        hoppings[1,2] = J_0_1 * np.exp(1j*np.vdot(A(time),a2))
        hoppings[1,0] = J_0_1 * np.exp(1j*np.vdot(A(time),-a2))
        hoppings[1,1] = J_0_0
        return hoppings
    N = 1
    n1 = np.linspace(-1,1,3,dtype='int')
    n2 = np.linspace(-1,1,3,dtype='int')

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

    # Along a2
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

    # enforcing reality of the blochvectors
    for k in range(3):
        for i in range(num_points):
                phi = 0.5*np.imag(np.log(np.inner(blochvectors1[i,:,k], 
                                                blochvectors1[i,:,k])))
                blochvectors1[i,:,k] = np.real(blochvectors1[i,:,k] * np.exp(-1j*phi))
    blochvectors1 = np.real(blochvectors1)
    for k in range(3):
        for i in range(num_points):
                phi = 0.5*np.imag(np.log(np.inner(blochvectors2[i,:,k], 
                                                blochvectors2[i,:,k])))
                blochvectors2[i,:,k] = np.real(blochvectors2[i,:,k] * np.exp(-1j*phi))
    blochvectors2 = np.real(blochvectors2)

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
    ax1.set_ylabel('$\epsilon T$')
    ax2.set_ylabel('$\epsilon T$')
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
        ax1.set_ylabel('$\epsilon T$')
        ax1.set_yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
                    3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                        '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
        ax1.set_xlabel('$k_x$')
        ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)

        positions = np.linspace(0,blochvectors1.shape[1] / 3-1,blochvectors1.shape[1] // 3)
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
        ax1.set_ylabel('$\epsilon T$')
        ax1.set_yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
                    3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                        '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
        ax1.set_xlabel('$k_y$')
        ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0','','$\pi$','','$2\pi$'])
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)

        positions = np.linspace(0,blochvectors2.shape[1] / 3-1,blochvectors2.shape[1] // 3)
        ax2.scatter(positions, loc)
        ax2.set_ylabel('amplitude')
        ax2.set_xlabel('position')
        plt.title('Cut along the y direction')
        plt.savefig('edge_state_localisation_a2/edge_state_localisation_{state}'.format(state=state))
        plt.close(fig)