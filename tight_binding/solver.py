import numpy as np
from .utilities import compute_reciprocal_lattice_vectors_2D
from .utilities import compute_time_evolution_operator
import matplotlib.pyplot as plt
from matplotlib import cm

class static_bandstructure2D:
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
                                dtype='complex')
        
        # Computing the right values for these grids
        for i in range(alpha_1.shape[0]):
            for j in range(alpha_1.shape[1]):
                k = np.array([kx[i,j], ky[i,j]])
                H = self.hamiltonian(k)
                eigenvalues, eigenvectors = np.linalg.eig(H)
                eigenvalues = np.real(eigenvalues) # H is hermitian
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
    
    def set_gauge(self):
        for band in range(self.bands):
            for i in range(self.blochvectors.shape[0]):
                for j in range(self.blochvectors.shape[1]):
                    if i == 0 and j == 0:
                        pass
                    elif i==0 and np.inner(self.blochvectors[i,j,band], 
                                           self.blochvectors[i,j-1,band]) < 0:
                        self.blochvectors[i,j,band] = -self.blochvectors[i,j,
                                                                         band]
                    elif np.inner(self.blochvectors[i,j,band], 
                                  self.blochvectors[i-1,j,band]) < 0:
                        self.blochvectors[i,j,band] = -self.blochvectors[i,j,
                                                                         band]
    
    def plot_blochvectors(self,
                          band,
                          component,
                          save,
                          kxmin=-np.pi,
                          kxmax=np.pi,
                          kymin=-np.pi,
                          kymax=np.pi):
        "Plots one component of the blochvectors."
        # Getting the correct region
        span = False
        reciprocal_vectors = 0
        while not span:
            reciprocal_vectors += 1
            beta_1 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_2 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_1, beta_2 = np.meshgrid(beta_1, beta_2, indexing='ij')
            kx = beta_1*self.b_1[0] + beta_2*self.b_2[0]
            ky = beta_1*self.b_1[1] + beta_2*self.b_2[1]
            span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
            
        # Specifying the indices of the required vectors in self.blochvectors
        i = (self.n*(beta_1%1)).astype(int)
        j = (self.n*(beta_2%1)).astype(int)
        blochvectors_expanded = self.blochvectors[i,j]
        blochvectors = np.transpose(blochvectors_expanded, (2,3,0,1))
        
        # Masking the data we do not want to plot
        blochvectors[:,
                     :,
                     (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(kx, ky, blochvectors[band,component], 
                               cmap=cm.coolwarm, linewidth=0)
        ax.set_xlim(kxmin,kxmax)
        ax.set_ylim(kymin,kymax)
        plt.savefig(save)
        plt.show()  

    def plot_bandstructure(self, 
                           save, 
                           kxmin=-np.pi, 
                           kxmax=np.pi, 
                           kymin=-np.pi, 
                           kymax=np.pi):
        "Plots and saves the bandstructure."
        # Getting the correct region
        span = False
        reciprocal_vectors = 0
        while not span:
            reciprocal_vectors += 1
            beta_1 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_2 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_1, beta_2 = np.meshgrid(beta_1, beta_2, indexing='ij')
            kx = beta_1*self.b_1[0] + beta_2*self.b_2[0]
            ky = beta_1*self.b_1[1] + beta_2*self.b_2[1]
            span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
            
        # Specifying the indices of the required energies in self.energies
        i = (self.n*(beta_1%1)).astype(int)
        j = (self.n*(beta_2%1)).astype(int)
        energies_expanded = self.energies[i,j]
        E = np.transpose(energies_expanded, (2,0,1))
        
        # Masking the data we do not want to plot
        E[:, (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.coolwarm,
                            linewidth=0)
        surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.coolwarm,
                            linewidth=0)
        surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.coolwarm,
                            linewidth=0)
        ax.set_xlim(kxmin,kxmax)
        ax.set_ylim(kymin,kymax)
        plt.savefig(save)
        plt.show()


class driven_bandstructure2D:
    """Contains all functionality to calculate the eigenvalues and -vectors."""
    def __init__(self,
        hamiltonian,
        period: float,
        dt: float,
        a_1: np.ndarray,
        a_2: np.ndarray,
        n: int,
        bands: int=3,
        lower_energy: float=-np.pi
    ):
        self.hamiltonian = hamiltonian
        self.a_1 = a_1
        self.a_2 = a_2
        self.b_1, self.b_2 = compute_reciprocal_lattice_vectors_2D(a_1,a_2)
        self.bands = bands
        self.n = n
        self.T = period
        self.dt = dt
        self.lower_energy = lower_energy

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
                                dtype='complex')
        
        # Computing the right values for these grids
        for i in range(alpha_1.shape[0]):
            for j in range(alpha_1.shape[1]):
                k = np.array([kx[i,j], ky[i,j]])
                def H(t):
                    return self.hamiltonian(k,t)
                U = compute_time_evolution_operator(H, self.T, self.dt)
                eigenvalues, eigenvectors = np.linalg.eig(U)
                eigenexp = np.real(np.log(eigenvalues) / (-1j))
                # converting to the right quasienergy range
                eigenexp = (eigenexp 
                            + 2*np.pi
                            *np.floor(((self.lower_energy/np.pi 
                                        - eigenexp/np.pi) / 2
                              + 1)))
                ind = np.argsort(eigenexp) #sorting the energies
                eigenexp = eigenexp[ind]
                eigenvectors = eigenvectors[:,ind]
                for band in range(len(eigenexp)):
                    energies[i,j,band] = eigenexp[band]
                    blochvectors[i,j,band] = eigenvectors[:,band]
        
        # Saving as class properties
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.kx = kx
        self.ky = ky
        self.energies = energies
        self.blochvectors = blochvectors

    def set_gauge(self):
        for band in range(self.bands):
            for i in range(self.blochvectors.shape[0]):
                for j in range(self.blochvectors.shape[1]):
                    if i == 0 and j == 0:
                        pass
                    elif i==0 and np.inner(self.blochvectors[i,j,band], 
                                           self.blochvectors[i,j-1,band]) < 0:
                        self.blochvectors[i,j,band] = -self.blochvectors[i,j,
                                                                         band]
                    elif np.inner(self.blochvectors[i,j,band], 
                                  self.blochvectors[i-1,j,band]) < 0:
                        self.blochvectors[i,j,band] = -self.blochvectors[i,j,
                                                                         band]

    def plot_blochvectors(self,
                          band,
                          component,
                          save,
                          kxmin=-np.pi,
                          kxmax=np.pi,
                          kymin=-np.pi,
                          kymax=np.pi):
        "Plots one component of the blochvectors."
        # Getting the correct region
        span = False
        reciprocal_vectors = 0
        while not span:
            reciprocal_vectors += 1
            beta_1 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_2 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_1, beta_2 = np.meshgrid(beta_1, beta_2, indexing='ij')
            kx = beta_1*self.b_1[0] + beta_2*self.b_2[0]
            ky = beta_1*self.b_1[1] + beta_2*self.b_2[1]
            span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
            
        # Specifying the indices of the required vectors in self.blochvectors
        i = (self.n*(beta_1%1)).astype(int)
        j = (self.n*(beta_2%1)).astype(int)
        blochvectors_expanded = self.blochvectors[i,j]
        blochvectors = np.transpose(blochvectors_expanded, (2,3,0,1))
        
        # Masking the data we do not want to plot
        blochvectors[:,
                     :,
                     (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(kx, ky, blochvectors[band,component], 
                               cmap=cm.coolwarm, linewidth=0)
        ax.set_xlim(kxmin,kxmax)
        ax.set_ylim(kymin,kymax)
        plt.savefig(save)
        plt.show()       
                                              
    def plot_bandstructure(self, 
                           save, 
                           kxmin=-np.pi, 
                           kxmax=np.pi, 
                           kymin=-np.pi, 
                           kymax=np.pi):
        "Plots and saves the bandstructure."
        # Getting the correct region
        span = False
        reciprocal_vectors = 0
        while not span:
            reciprocal_vectors += 1
            beta_1 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_2 = np.linspace(-reciprocal_vectors,
                                 reciprocal_vectors,
                                 2*reciprocal_vectors*self.n)
            beta_1, beta_2 = np.meshgrid(beta_1, beta_2, indexing='ij')
            kx = beta_1*self.b_1[0] + beta_2*self.b_2[0]
            ky = beta_1*self.b_1[1] + beta_2*self.b_2[1]
            span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
        # Specifying the indices of the required energies in self.energies
        i = (self.n*(beta_1%1)).astype(int)
        j = (self.n*(beta_2%1)).astype(int)
        energies_expanded = self.energies[i,j]
        E = np.transpose(energies_expanded, (2,0,1))
        
        # Masking the data we do not want to plot
        E[:, (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.coolwarm,
                            linewidth=0)
        surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.coolwarm,
                            linewidth=0)
        surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.coolwarm,
                            linewidth=0)
        ax.set_xlim(kxmin,kxmax)
        ax.set_ylim(kymin,kymax)
        plt.savefig(save)
        plt.show()