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


# PARAMETERS
num_points = 200
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


