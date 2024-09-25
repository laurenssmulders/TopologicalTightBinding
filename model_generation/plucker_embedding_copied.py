import numpy as np
import matplotlib.pyplot as plt

from tight_binding.bandstructure import plot_bandstructure2D, sort_energy_grid
from tight_binding.topology import gauge_fix_grid

N = 2
NUM_POINTS = 100
r = 1
c = 1
a1 = np.array([1,0])
a2 = np.array([0,1])

# HOPPINGS
t1 = np.array(
    [
        [0.00442999 -0.00756786*1j,-0.0380643+0.0154554*1j,
         -0.00125-0.00378393*1j,0.0405643 -0.0078875*1j,
         -0.00692999+1.66533e-18*1j],
        [-0.0380643+0.0154554*1j,-0.0602567-0.0233429*1j,
         0.00125 +0.0116714*1j,0.0577567 -1.11022e-18*1j,
         0.0405643 +0.0078875*1j],
        [-0.00125-0.00378393*1j,0.00125 +0.0116714*1j,-0.00125+0.*1j,
         0.00125 -0.0116714*1j,-0.00125+0.00378393*1j],
        [0.0405643 -0.0078875*1j,0.0577567 +1.11022e-18*1j,
         0.00125 -0.0116714*1j,-0.0602567+0.0233429*1j,
         -0.0380643-0.0154554*1j],
        [-0.00692999-1.66533e-18*1j,0.0405643 +0.0078875*1j,
         -0.00125+0.00378393*1j,-0.0380643-0.0154554*1j,
         0.00442999 +0.00756786*1j]
    ])

t3 = np.array(
    [
        [-0.00125-4.23273e-18*1j,-0.04417-5.55112e-19*1j,
         -0.086332+3.33067e-18*1j,-0.04417+5.55112e-18*1j,
         -0.00125-1.17267e-17*1j],
        [0.04167 +2.41127e-18*1j,-0.00125+4.996e-18*1j,
         0.0187531 +7.21645e-18*1j,-0.00125+6.66134e-18*1j,
         0.04167 +1.16226e-18*1j],
        [0.083832 -8.10052e-19*1j,-0.0212531-4.02966e-18*1j,-0.00125+0.*1j,
         -0.0212531+4.02966e-18*1j,0.083832 +8.10052e-19*1j],
        [0.04167 -1.16226e-18*1j,-0.00125-6.66134e-18*1j,
         0.0187531 -7.21645e-18*1j,-0.00125-4.996e-18*1j,
         0.04167 -2.41127e-18*1j],
        [-0.00125+1.17267e-17*1j,-0.04417-5.55112e-18*1j,
         -0.086332-3.33067e-18*1j,-0.04417+5.55112e-19*1j,
         -0.00125+4.23273e-18*1j]
    ])

t4 = np.array(
    [
        [9.99201e-18+0.0138766*1j,-2.9976e-17+0.113774*1j,
         4.44089e-18+0.245873*1j,-2.498e-17+0.113774*1j,
         -1.77636e-17+0.0138766*1j],
        [-7.35523e-18+0.0317923*1j,-1.66533e-18+0.0570757*1j,0. -0.140491*1j,
         -3.33067e-18+0.0570757*1j,1.52656e-18+0.0317923*1j],
        [4.16334e-18-1.97758e-18*1j,5.55112e-19-2.81025e-18*1j,
         -3.10862e-17+0.*1j,5.55112e-19+2.81025e-18*1j,
         4.16334e-18+1.97758e-18*1j],
        [1.52656e-18-0.0317923*1j,-3.33067e-18-0.0570757*1j,0. +0.140491*1j,
         -1.66533e-18-0.0570757*1j,-7.35523e-18-0.0317923*1j],
        [-1.77636e-17-0.0138766*1j,-2.498e-17-0.113774*1j,
         4.44089e-18-0.245873*1j,-2.9976e-17-0.113774*1j,
         9.99201e-18-0.0138766*1j]
    ])

t6 = np.array(
    [
        [8.04912e-18+0.0138766*1j,-3.33067e-18+0.0317923*1j,
         -3.66097e-18+1.11022e-18*1j,-3.33067e-18-0.0317923*1j,
         -1.88738e-17-0.0138766*1j],
        [-3.42781e-17+0.113774*1j,1.11022e-18+0.0570757*1j,
         4.14669e-18-1.17961e-18*1j,-2.77556e-18-0.0570757*1j,
         -1.72085e-17-0.113774*1j],
        [-4.14669e-18+0.245873*1j,-9.86987e-18-0.140491*1j,-3.4528e-18+0.*1j,
         -9.86987e-18+0.140491*1j,-4.14669e-18-0.245873*1j],
        [-1.72085e-17+0.113774*1j,-2.77556e-18+0.0570757*1j,
         4.14669e-18+1.17961e-18*1j,1.11022e-18-0.0570757*1j,
         -3.42781e-17-0.113774*1j],
        [-1.88738e-17+0.0138766*1j,-3.33067e-18+0.0317923*1j,
         -3.66097e-18-1.11022e-18*1j,-3.33067e-18-0.0317923*1j,
         8.04912e-18-0.0138766*1j]
    ])

t8 = np.array(
    [
        [1.27676e-17+3.10862e-17*1j,-0.0939283+1.55431e-17*1j,
         -0.216506+4.44089e-18*1j,-0.0939283-5.37764e-18*1j,
         6.66134e-18-9.71445e-18*1j],
        [-0.0939283-3.60822e-18*1j,1.83187e-17+1.85962e-17*1j,
         0.154125 +2.88658e-17*1j,8.60423e-18+1.31839e-17*1j,
         -0.0939283+1.59595e-17*1j],
        [-0.216506-7.35523e-18*1j,0.154125 +1.42247e-17*1j,-8.88178e-18+0.*1j,
         0.154125 -1.42247e-17*1j,-0.216506+7.35523e-18*1j],
        [-0.0939283-1.59595e-17*1j,8.60423e-18-1.31839e-17*1j,
         0.154125 -2.88658e-17*1j,1.83187e-17-1.85962e-17*1j,
         -0.0939283+3.60822e-18*1j],
        [6.66134e-18+9.71445e-18*1j,-0.0939283+5.37764e-18*1j,
         -0.216506-4.44089e-18*1j,-0.0939283-1.55431e-17*1j,
         1.27676e-17-3.10862e-17*1j]
    ])

# GELL MANN MATRICES
GM1 = np.array([
    [0,1,0],
    [1,0,0],
    [0,0,0]
])

GM3 = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,0]
])

GM4 = np.array([
    [0,0,1],
    [0,0,0],
    [1,0,0]
])

GM6 = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0]
])

GM8 = 1 / (3**0.5) * np.array([
    [1,0,0],
    [0,1,0],
    [0,0,-2]
])

# FINAL HOPPINGS
J = np.zeros((2*N+1, 2*N+1, 3, 3), dtype='complex')
for i in range(2*N+1):
    for j in range(2*N+1):
        J[i,j]  = (t1[i,j]*GM1 + t3[i,j]*GM3 + t4[i,j]*GM4 + t6[i,j]*GM6 
                   + t8[i,j]*GM8)
        
np.save('J.npy', J)

#Reality of hoppings
imaginary_fraction = np.sum(np.abs(np.imag(J))) / np.sum(np.abs(J))
print('Imaginary fraction of the hoppings: ', imaginary_fraction)

# GOING BACKWARDS
# CALCULATING THE HAMILTONIAN
print('Calculating the hamiltonian backwards...')
k = np.linspace(0, 2*np.pi, NUM_POINTS)
kx, ky = np.meshgrid(k,k,indexing='ij')
hoppings = J
hoppings = hoppings[np.newaxis,np.newaxis,:,:,:,:]
hoppings = np.repeat(hoppings,NUM_POINTS,0)
hoppings = np.repeat(hoppings,NUM_POINTS,1)

hamiltonian = np.zeros((NUM_POINTS,NUM_POINTS,3,3),dtype='complex')
n1 = np.linspace(-N,N,2*N+1,dtype='int')
n2 = np.linspace(-N,N,2*N+1,dtype='int')
kx = kx[:,:,np.newaxis,np.newaxis]
kx = np.repeat(kx,3,2)
kx = np.repeat(kx,3,3)
ky = ky[:,:,np.newaxis,np.newaxis]
ky = np.repeat(ky,3,2)
ky = np.repeat(ky,3,3)

for i in range(2*N+1):
    for j in range(2*N+1):
        exponent = np.exp(1j*(n1[i]*kx+n2[j]*ky))
        hamiltonian -= hoppings[:,:,i,j,:,:] * exponent

## Checking hermiticity
hermiticity_error = (np.sum(np.abs(hamiltonian 
                                  - np.conjugate(np.transpose(hamiltonian, 
                                                              (0,1,3,2))))) 
                                                              / NUM_POINTS**2)
print('Hermiticity error: ', hermiticity_error)

## Checcking reality
reality_error = np.sum(np.abs(np.imag(hamiltonian))) / NUM_POINTS**2
print('Reality error: ', reality_error)

# Diagonalising
print('Diagonalising...')
eigenvalues, blochvectors = np.linalg.eig(hamiltonian)
energies = np.real(eigenvalues)

# enforcing reality of the blochvectors
for k in range(3):
    for i in range(NUM_POINTS):
        for j in range(NUM_POINTS):
            phi = 0.5*np.imag(np.log(np.inner(blochvectors[i,j,:,k], 
                                            blochvectors[i,j,:,k])))
            blochvectors[i,j,:,k] = np.real(blochvectors[i,j,:,k] * np.exp(-1j*phi))
blochvectors = np.real(blochvectors)

energies, blochvectors = sort_energy_grid(energies,blochvectors,regime='static')

# plotting
print('Plotting...')
plot_bandstructure2D(energies,a1,a2,'test.png',r=r,c=c,regime='static')

np.save('blochvectors.npy', blochvectors)

blochvectors = gauge_fix_grid(blochvectors)
k = np.linspace(0,1,NUM_POINTS,endpoint=False)
kx, ky = np.meshgrid(k,k,indexing='ij')
u = blochvectors[:,:,0,0]
v = blochvectors[:,:,1,0]
plt.quiver(kx,ky,u,v, width=0.001)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
