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

def square_hamiltonian_static(delta_a, 
                              delta_b, 
                              delta_c, 
                              delta_ab, 
                              delta_ac, 
                              delta_bc, 
                              J_a, 
                              J_b, 
                              J_c, 
                              J_ab, 
                              J_ac, 
                              J_bc, 
                              a):
    """Defines a bloch hamiltonian for the equillibrium square lattice.
    
    Parameters
    ----------
    delta_a: float
        The on-site potential for the A orbital
    delta_b: float
        The on-site potential for the B orbital
    delta_c: float
        The on-site potential for the C orbital
    delta_ab: float
        The matrix element for the A and B orbitals at the same site
    delta_ac: float
        The matrix element for the A and C orbitals at the same site
    delta_bc: float
        The matrix element for the B and C orbitals at the same site
    J_a: float
        The tunneling parameter between nn A orbitals
    J_b: float
        The tunneling parameter between nn B orbitals
    J_b: float
        The tunneling parameter between nn C orbitals
    J_ab: float
        The tunneling parameter between nn A and B orbitals
    J_ac: float
        The tunneling parameter between nn A and C orbitals
    J_bc: float
        The tunneling parameter between nn B and C orbitals
    a: float
        THe lattice spacing

    Returns
    -------
    H: function
        The square bloch hamiltonian as a function of quasimomentum k.
    """
    a_1 = a*np.transpose(np.array([[1,0]]))
    a_2 = a*np.transpose(np.array([[0,1]]))

    delta_ba = np.conjugate(delta_ab)
    delta_ca = np.conjugate(delta_ac)
    delta_cb = np.conjugate(delta_bc)
    J_ba = np.conjugate(J_ab)
    J_ca = np.conjugate(J_ac)
    J_cb = np.conjugate(J_bc)

    def H(k):
       c = np.cos(np.vdot(k,a_1)) + np.cos(np.vdot(k,a_2))
       hamiltonian = np.array([
           [delta_a - 2*c*J_a, delta_ab - 2*c*J_ab, delta_ac - 2*c*J_ac],
           [delta_ba - 2*c*J_ba, delta_b - 2*c*J_b, delta_bc - 2*c*J_bc],
           [delta_ca - 2*c*J_ca, delta_cb - 2*c*J_cb, delta_c - 2*c*J_c]
       ])
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

def square_hamiltonian_driven(delta_a, 
                              delta_b, 
                              delta_c, 
                              delta_ab, 
                              delta_ac, 
                              delta_bc, 
                              J_a, 
                              J_b, 
                              J_c, 
                              J_ab, 
                              J_ac, 
                              J_bc, 
                              a, 
                              A_x, 
                              A_y, 
                              omega, 
                              phi):
    """Defines a bloch hamiltonian for the out-of-equillibrium square lattice.
    
    Parameters
    ----------
    delta_a: float
        The on-site potential for the A orbital
    delta_b: float
        The on-site potential for the B orbital
    delta_c: float
        The on-site potential for the C orbital
    delta_ab: float
        The matrix element for the A and B orbitals at the same site
    delta_ac: float
        The matrix element for the A and C orbitals at the same site
    delta_bc: float
        The matrix element for the B and C orbitals at the same site
    J_a: float
        The tunneling parameter between nn A orbitals
    J_b: float
        The tunneling parameter between nn B orbitals
    J_b: float
        The tunneling parameter between nn C orbitals
    J_ab: float
        The tunneling parameter between nn A and B orbitals
    J_ac: float
        The tunneling parameter between nn A and C orbitals
    J_bc: float
        The tunneling parameter between nn B and C orbitals
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
        The square bloch hamiltonian as a function of quasimomentum k and 
        time t"""
    a_1 = a*np.transpose(np.array([[1,0]]))
    a_2 = a*np.transpose(np.array([[0,1]]))

    delta_ba = np.conjugate(delta_ab)
    delta_ca = np.conjugate(delta_ac)
    delta_cb = np.conjugate(delta_bc)
    J_ba = np.conjugate(J_ab)
    J_ca = np.conjugate(J_ac)
    J_cb = np.conjugate(J_bc)

    def A(time):
        drive = np.transpose(np.array([[A_x*np.cos(omega*time), 
                                        -A_y*np.cos(omega*time + phi)]]))
        return drive
    
    def H(k,t):
       c = np.cos(np.vdot(k + A(t),a_1)) + np.cos(np.vdot(k + A(t),a_2))
       hamiltonian = np.array([
           [delta_a - 2*c*J_a, delta_ab - 2*c*J_ab, delta_ac - 2*c*J_ac],
           [delta_ba - 2*c*J_ba, delta_b - 2*c*J_b, delta_bc - 2*c*J_bc],
           [delta_ca - 2*c*J_ca, delta_cb - 2*c*J_cb, delta_c - 2*c*J_c]
       ])
       return hamiltonian
    return H