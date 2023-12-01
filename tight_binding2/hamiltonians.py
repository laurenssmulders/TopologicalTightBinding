"""This package contains defines Bloch hamiltonians for different lattices."""


# 01 IMPORTS

import numpy as np

# 02 EQUILLIBRIUM HAMILTONIANS

def kagome_hamiltonian_static(delta_a, delta_b, delta_c, J, a):
    """Defines a bloch hamiltonian for the equillibrium kagome lattice.
    
    Parameters
    ----------
    delta_a: float
        The on-site potential for the A site
    delta_b: float
        The on-site potential for the B site
    delta_c: float
        The on-site potential for the C site
    J: float
        The tunneling parameter
    a: float
        THe lattice spacing

    Returns
    -------
    H: function
        The kagome bloch hamiltonian as a function of quasimomentum k.
    """
    d_ba = a*np.transpose(np.array([[0.5, 0]]))
    d_ca = a*np.transpose(np.array([[0.25, 0.25*3**0.5]]))
    d_cb = a*np.transpose(np.array([[-0.25, 0.25*3**0.5]]))
    d_ab = -d_ba
    d_ac = -d_ca
    d_bc = -d_cb

    def H(k):
        hamiltonian = np.array([[delta_a, 
                                 -2*J*np.cos(np.vdot(k,d_ba)), 
                                 -2*J*np.cos(np.vdot(k,d_ca))],
                                [-2*J*np.cos(np.vdot(k,d_ab)), 
                                 delta_b, 
                                 -2*J*np.cos(np.vdot(k,d_cb))],
                                [-2*J*np.cos(np.vdot(k,d_ac)), 
                                 -2*J*np.cos(np.vdot(k,d_bc)), 
                                 delta_c]])
        return hamiltonian
    return H

# 03 OUT-OF-EQUILLIBRIUM HAMILTONIANS

def kagome_hamiltonian_driven(delta_a, 
                              delta_b, 
                              delta_c, 
                              J, 
                              a, 
                              A_x, 
                              A_y, 
                              omega, 
                              phi):
    """Defines a bloch hamiltonian for the out-of-equillibrium kagome lattice.
    
    Parameters
    ----------
    delta_a: float
        The on-site potential for the A site
    delta_b: float
        The on-site potential for the B site
    delta_c: float
        The on-site potential for the C site
    J: float
        The tunneling parameter
    a: float
        The lattice spacing
    A_x: float
        The amplitude of the x component of the driving vector potential
    A_y: float
        The amplitude of the y component of the driving vector potential
    omega: float
        The angular frequency of the driving vector potential
    phi:
        The relative phase of the phi component of the vector potential

    Returns
    -------
    H: function
        The kagome bloch hamiltonian as a function of quasimomentum k and 
        time t"""
    d_ba = a*np.transpose(np.array([[0.5, 0]]))
    d_ca = a*np.transpose(np.array([[0.25, 0.25*3**0.5]]))
    d_cb = a*np.transpose(np.array([[-0.25, 0.25*3**0.5]]))
    d_ab = -d_ba
    d_ac = -d_ca
    d_bc = -d_cb

    def A(time):
        drive = np.transpose(np.array([[A_x*np.cos(omega*time), 
                                        -A_y*np.cos(omega*time + phi)]]))
        return drive
    
    def H(k,t):
        hamiltonian = np.array([[delta_a, 
                                 -2*J*np.cos(np.vdot(k + A(t),d_ba)), 
                                 -2*J*np.cos(np.vdot(k + A(t),d_ca))],
                                [-2*J*np.cos(np.vdot(k + A(t),d_ab)), 
                                 delta_b, 
                                 -2*J*np.cos(np.vdot(k + A(t),d_cb))],
                                [-2*J*np.cos(np.vdot(k + A(t),d_ac)), 
                                 -2*J*np.cos(np.vdot(k + A(t),d_bc)), 
                                 delta_c]]) 
        return hamiltonian
    return H