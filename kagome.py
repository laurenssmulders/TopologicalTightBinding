import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tight_binding.topology import compute_patch_euler_class, gauge_fix_grid, gauge_fix_path
from tight_binding.hamiltonians import kagome_hamiltonian_static
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.diagonalise import compute_eigenstates#
from tight_binding.utilitities import compute_reciprocal_lattice_vectors_2D

H = kagome_hamiltonian_static(0,0,0,1,1)
a_1 = np.array([1,0])
a_2 = 0.5*np.array([1,3**0.5])

energies, vectors = compute_bandstructure2D(H,a_1,a_2,100,regime='static')

b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)

# Creating a grid of coefficients for the reciprocal lattice vectors
alpha = np.linspace(0,1,100,endpoint=False)
alpha_1, alpha_2 = np.meshgrid(alpha, alpha, indexing='ij')

# Finding the corresponding k_vectors
kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(kx, ky, vectors[:,:,2,0], cmap=cm.YlGnBu, linewidth=0)
plt.show()
plt.close()

check = np.zeros(vectors.shape[:2])
for i in range(check.shape[0]):
    for j in range(check.shape[1]):
        k = np.array([kx[i,j],ky[i,j]])
        check[i,j] = np.vdot(vectors[i,j,:,0], np.dot(H(k),vectors[i,j,:,0])) / energies[i,j,0]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(kx, ky, check, cmap=cm.YlGnBu, linewidth=0)
plt.show()

vectors_gf = gauge_fix_grid(vectors)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(kx, ky, vectors_gf[:,:,2,0], cmap=cm.YlGnBu, linewidth=0)
plt.show()

#chi = compute_patch_euler_class(-1,1,-1,1,[1,2],H,100,0,0,0,0,'static')
#print(chi)