import numpy as np
from utilities import compute_reciprocal_lattice_vectors_2D
import matplotlib.pyplot as plt
from matplotlib import cm

class bandstructure2D:
    """Contains all functionality to calculate the eigenvalues and -vectors."""
    def __init__(self,
        hamiltonian,
        a_1: np.ndarray,
        a_2: np.ndarray,
        bands: int = 3 
    ):
        self.hamiltonian = hamiltonian
        self.a_1 = a_1
        self.a_2 = a_2
        self.b_1, self.b_2 = compute_reciprocal_lattice_vectors_2D(a_1,a_2)
        self.bands = bands
    
    
    def create_grid(self, dx: float, xspan: float, yspan: float):
        "Creates square grid centred on origin"
        grid = np.zeros((int(xspan / dx), int(yspan / dx), 2), dtype='float')
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid[i,j] = np.array([(-xspan + dx) / 2 + i*dx, 
                                      (-yspan + dx) / 2 + j*dx])
        self.grid = grid

    def compute_bandstructure(self):
        "Computes the bandstructure and bloch states on the grid"
        self.bandstructure = np.zeros([self.grid.shape[0], 
                                       self.grid.shape[1], 
                                       self.bands], dtype=float)
        self.blochvectors = np.zeros([self.grid.shape[0], 
                                       self.grid.shape[1], 
                                       self.bands,
                                       self.bands], dtype=float)
                    
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                k = self.grid[i,j]
                H = self.hamiltonian(k)
                eigenvalues, eigenvectors = np.linalg.eig(H)
                ind = np.argsort(eigenvalues) #sorting the energies
                eigenvalues = eigenvalues[ind]
                eigenvectors = eigenvectors[:,ind]
                for eigenvalue in range(len(eigenvalues)):
                    self.bandstructure[i,j,eigenvalue] = eigenvalues[eigenvalue]
                    self.blochvectors[i,j,eigenvalue] = eigenvectors[:,
                                                                     eigenvalue]
    
    def plot_bandstructure(self, save):
        "Plots and saves the bandstructure." 
        kx = self.grid[:,:,0]
        ky = self.grid[:,:,1]
        E = np.transpose(self.bandstructure, (2,0,1))
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        plt.savefig(save)
        plt.show()
        