"""Module with functions to calculate the generalised eigenstates and 
    (quasi-)energies of a Bloch hamiltonian.
"""

# 01 IMPORTS
import numpy as np
import scipy.linalg as la

def compute_time_evolution(hamiltonian, k, t_0, t_1, num_steps, method='trotter'):
    """Computes the time evolution operator after a certain time.
    
    Parameters
    ----------
    hamiltonian: function
        The bloch hamiltonian as a function of k and t
    k: numpy.ndarray
        The quasimomentum at which to evaluate the bloch hamiltonian
    t_0: float
        The time for the initial state
    t_1: float
        The time for the final state
    num_steps: int
        The number of steps to take in calculating U
    method: float
        The method with which to solve Schrodinger's equation. Trotter or 
        Runge-Kutta 4
    
    Returns
    -------
    U: np.ndarray
        The time evolution operator from t_0 to t_1.
    """
    def H(t):
        return hamiltonian(k,t)
    
    if method == 'trotter':
        times = np.linspace(t_0,t_1,num_steps,endpoint=False)
        dt = (t_1 - t_0) / num_steps
        U = np.identity(H(t_0).shape[0])
        for i in range(len(times)):
            U = np.dot(la.expm(-1j * H(times[i]) * dt), U)
    
    elif method == 'Runge-Kutta':
        times = np.linspace(t_0,t_1,num_steps, endpoint=False)
        dt = (t_1 - t_0) / num_steps
        U = np.identity(H(t_0).shape[0])
        def f(time, operator):
            return -1j*np.dot(H(time),operator)
        for i in range(len(times)):
            k_1 = f(times[i],U)
            k_2 = f(times[i] + dt / 2, U + dt * k_1 / 2)
            k_3 = f(times[i] + dt / 2, U + dt * k_2 / 2)
            k_4 = f(times[i] + dt, U + dt * k_3)
            U = U + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
    
    else:
        print('Enter valid solution method.')
    
    #Checking for unitarity
    norm = np.dot(np.conjugate(np.transpose(U, (1,0))), U)
    norm_error = np.trace(norm) - U.shape[0]
    if abs(norm_error) > 1e-5:
        print('High normalisation error!: {norm_error}'.format(
            norm_error=norm_error))
    
    return U 

def compute_eigenstates(hamiltonian, 
                        k, 
                        omega=0, 
                        num_steps=0,
                        lowest_quasi_energy=-np.pi,
                        method='trotter', 
                        regime='driven'):
    """Computes the eigenstates and energies for a static or driven bloch 
    hamiltonian at a certain quasimomentum k.
    
    Parameters
    ----------
    hamiltonian: function
        The bloch hamiltonian for which to find the eigenstates
    k: np.ndarray
        The quasimomentum at which to evaluate the bloch hamiltonian
    omega: float
        The angular frequency of the bloch hamiltonian in case of a driven 
        system
    num_steps: int
        The number of steps to use in the calculation of the time evolution
    lowest_quasi_energy: float
        The lower bound of the 2pi interval in which to give the quasi energies
    method: str
        The method for calculating the time evolution: trotter or Runge-Kutta
    regime: str
        'driven' or 'static'
    
    Returns
    -------
    energies: np.ndarray
        An array with the eigenenergies, sorted from low to high.
    blochvectors: np.ndarray
        An array where the blochvectors[:,i] is the blochvector corresponding to 
        energies[i]
    """
    if regime == 'static':
        H = hamiltonian(k)
        energies, blochvectors = np.linalg.eig(H)
    
    elif regime == 'driven':
        U = compute_time_evolution(hamiltonian, k, 0, 2*np.pi/omega, num_steps, 
                                   method)
        eigenvalues, eigenvectors = np.linalg.eig(U)
        energies = np.real(np.log(eigenvalues) / (-1j))
        errors = np.real(np.log(eigenvalues)) #checking for real eigenenergies
        if np.sum(errors) > 1e-5:
            print('Imaginary quasienergies!')
        #Getting the energies in the right range
        energies = (energies + 2*np.pi*np.floor((lowest_quasi_energy-energies) 
                                                / (2*np.pi) + 1))
        blochvectors = eigenvectors

    else:
        print('Enter a valid regime')
    
    #Sorting the energies and blochvectors
    ind = np.argsort(energies)
    energies = energies[ind]
    blochvectors = blochvectors[:,ind]
    return energies, blochvectors