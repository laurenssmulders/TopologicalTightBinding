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
from matplotlib import cm

from tight_binding.bandstructure import sort_energy_grid, plot_bandstructure2D
from tight_binding.utilities import compute_reciprocal_lattice_vectors_2D
from tight_binding.topology import gauge_fix_grid


# PARAMETERS
plot = True
euler = False

num_points = 100
num_steps = 100
lowest_quasi_energy = -np.pi
r = 1
c = 1

omega = 20
dJ1x = 0
dJ1y = 0
dJ2 = 0
dA = 0
dC = 0

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

    plot_bandstructure2D(energies, a1, a2, 'test.png', lowest_quasi_energy=lowest_quasi_energy,r=r,c=c)

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
