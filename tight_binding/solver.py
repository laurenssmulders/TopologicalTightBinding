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
        n: int,
        bands: int=3  
    ):
        self.hamiltonian = hamiltonian
        self.a_1 = a_1
        self.a_2 = a_2
        self.b_1, self.b_2 = compute_reciprocal_lattice_vectors_2D(a_1,a_2)
        self.bands = bands
        self.n = n

    def compute_bandstructure(self):
        """Computes the bandstructure for one reciprocal unit cell."""
        # Creating the required grids mapping from points in reciprocal space
        alpha_1 = np.linspace(0,1,self.n)
        alpha_2 = np.linspace(0,1,self.n)
        alpha_1, alpha_2 = np.meshgrid(alpha_1, alpha_2, indexing='ij')
        kx = alpha_1*self.b_1[0] + alpha_2*self.b_2[0]
        ky = alpha_1*self.b_1[1] + alpha_2*self.b_2[1]
        energies = np.zeros(alpha_1.shape + (self.bands,), dtype='float')
        blochvectors = np.zeros(alpha_1.shape + (self.bands,self.bands), 
                                dtype='float')
        
        # Computing the right values for these grids
        for i in range(alpha_1.shape[0]):
            for j in range(alpha_1.shape[1]):
                k = np.array([kx[i,j], ky[i,j]])
                H = self.hamiltonian(k)
                eigenvalues, eigenvectors = np.linalg.eig(H)
                ind = np.argsort(eigenvalues) #sorting the energies
                eigenvalues = eigenvalues[ind]
                eigenvectors = eigenvectors[:,ind]
                for band in range(len(eigenvalues)):
                    energies[i,j,band] = eigenvalues[band]
                    blochvectors[i,j,band] = eigenvectors[:,band]
        
        # Saving as class properties
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.kx = kx
        self.ky = ky
        self.energies = energies
        self.blochvectors = blochvectors
    
    def plot_gauge(self, band_index, save):
        """Visualises the gauge continuity over the BZ"""
        gauge = np.zeros(self.alpha_1.shape, dtype='float')
        for i in range(gauge.shape[0]):
            for j in range(gauge.shape[1]):
                gauge_indicator = (
                    np.inner(self.blochvectors[i,j,band_index], 
                            self.blochvectors[(i+1) % self.blochvectors.shape[0], 
                                              j, 
                                              band_index])
                    + np.inner(self.blochvectors[i,j,band_index], 
                            self.blochvectors[(i-1) % self.blochvectors.shape[0], 
                                              j, 
                                              band_index])
                    + np.inner(self.blochvectors[i,j,band_index], 
                            self.blochvectors[i, 
                                              (j+1) % self.blochvectors.shape[1], 
                                              band_index])
                    + np.inner(self.blochvectors[i,j,band_index], 
                            self.blochvectors[i, 
                                              (j-1) % self.blochvectors.shape[1], 
                                              band_index])) / 4
                gauge[i,j] = gauge_indicator
        
                    
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(self.kx, self.ky, gauge, cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)
        plt.savefig(save)
        plt.show()        
                                              



    def plot_bandstructure(self, save):
        "Plots and saves the bandstructure." 
        E = np.transpose(self.energies, (2,0,1))
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf1 = ax.plot_surface(self.kx, self.ky, E[0], cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)
        surf2 = ax.plot_surface(self.kx, self.ky, E[1], cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)
        surf3 = ax.plot_surface(self.kx, self.ky, E[2], cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)
        plt.savefig(save)
        plt.show()
        