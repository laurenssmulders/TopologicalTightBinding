"""This package defines Bloch hamiltonians for different lattices."""


# 01 IMPORTS

import numpy as np
from tight_binding.utilities import gell_mann

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
    d_ba = a*np.array([0.5, 0])
    d_ca = a*np.array([0.25, 0.25*3**0.5])
    d_cb = a*np.array([-0.25, 0.25*3**0.5])
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

def square_hamiltonian_static(delta_a=0, 
                              delta_b=0, 
                              delta_c=0, 
                              J_ab_0=0, 
                              J_ac_0=0, 
                              J_bc_0=0, 
                              J_aa_1x=0, 
                              J_bb_1x=0, 
                              J_cc_1x=0, 
                              J_ab_1x=0, 
                              J_ac_1x=0, 
                              J_bc_1x=0,
                              J_aa_1y=0, 
                              J_bb_1y=0, 
                              J_cc_1y=0, 
                              J_ab_1y=0, 
                              J_ac_1y=0, 
                              J_bc_1y=0,
                              J_aa_2p=0, 
                              J_bb_2p=0, 
                              J_cc_2p=0, 
                              J_ab_2p=0, 
                              J_ac_2p=0, 
                              J_bc_2p=0,
                              J_aa_2m=0, 
                              J_bb_2m=0, 
                              J_cc_2m=0, 
                              J_ab_2m=0, 
                              J_ac_2m=0, 
                              J_bc_2m=0,
                              a=1):
    """Defines a bloch hamiltonian for the out-of-equillibrium square lattice.
    
    Parameters
    ----------
    delta_aa: float
        The on-site potential for the A orbital
    delta_bb: float
        The on-site potential for the B orbital
    delta_cc: float
        The on-site potential for the C orbital
    delta_ab: float
        The matrix element for the A and B orbitals at the same site
    delta_ac: float
        The matrix element for the A and C orbitals at the same site
    delta_bc: float
        The matrix element for the B and C orbitals at the same site
    J_aa_1x: float
        The tunneling parameter between nn A orbitals in the horizontal 
        direction
    J_bb_1x: float
        The tunneling parameter between nn B orbitals in the horizontal 
        direction
    J_cc_1x: float
        The tunneling parameter between nn C orbitals in the horizontal 
        direction
    J_ab_1x: float
        The tunneling parameter between nn A and B orbitals in the horizontal 
        direction
    J_ac_1x: float
        The tunneling parameter between nn A and C orbitals in the horizontal 
        direction
    J_bc_1x: float
        The tunneling parameter between nn B and C orbitals in the horizontal 
        direction
    J_aa_1y: float
        The tunneling parameter between nn A orbitals in the vertical direction
    J_bb_1y: float
        The tunneling parameter between nn B orbitals in the vertical direction
    J_cc_1y: float
        The tunneling parameter between nn C orbitals in the vertical direction
    J_ab_1y: float
        The tunneling parameter between nn A and B orbitals in the vertical 
        direction
    J_ac_1y: float
        The tunneling parameter between nn A and C orbitals in the vertical 
        direction
    J_bc_1y: float
        The tunneling parameter between nn B and C orbitals in the vertical 
        direction
    J_aa_2p: float
        The tunneling parameter between nn A orbitals in the diagonal a1+a2 
        direction
    J_bb_2p: float
        The tunneling parameter between nn B orbitals in the diagonal a1+a2 
        direction
    J_cc_2p: float
        The tunneling parameter between nn C orbitals in the diagonal a1+a2 
        direction
    J_ab_2p: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1+a2 direction
    J_ac_2p: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1+a2 direction
    J_bc_2p: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1+a2 direction
    J_aa_2m: float
        The tunneling parameter between nn A orbitals in the diagonal a1-a2 
        direction
    J_bb_2m: float
        The tunneling parameter between nn B orbitals in the diagonal a1-a2 
        direction
    J_cc_2m: float
        The tunneling parameter between nn C orbitals in the diagonal a1-a2 
        direction
    J_ab_2m: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1-a2 direction
    J_ac_2m: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1-a2 direction
    J_bc_2m: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1-a2 direction
    a: float
        The lattice spacing

    Returns
    -------
    H: function
        The square bloch hamiltonian as a function of quasimomentum k and 
        time t"""
    a_1 = a*np.array([1,0])
    a_2 = a*np.array([0,1])

    J_ba_0 = np.conjugate(J_ab_0)
    J_ca_0 = np.conjugate(J_ac_0)
    J_cb_0 = np.conjugate(J_bc_0)
    J_ba_1x = np.conjugate(J_ab_1x)
    J_ca_1x = np.conjugate(J_ac_1x)
    J_cb_1x = np.conjugate(J_bc_1x)
    J_ba_1y = np.conjugate(J_ab_1y)
    J_ca_1y = np.conjugate(J_ac_1y)
    J_cb_1y = np.conjugate(J_bc_1y)
    J_ba_2p = np.conjugate(J_ab_2p)
    J_ca_2p = np.conjugate(J_ac_2p)
    J_cb_2p = np.conjugate(J_bc_2p)
    J_ba_2m = np.conjugate(J_ab_2m)
    J_ca_2m = np.conjugate(J_ac_2m)
    J_cb_2m = np.conjugate(J_bc_2m)
    
    def H(k):
       H_offsets = np.array([
           [delta_a, 0, 0],
           [0, delta_b, 0],
           [0, 0, delta_c]
       ])

       H_onsite = np.array([
           [0, J_ab_0, J_ac_0],
           [J_ba_0, 0, J_bc_0],
           [J_ca_0, J_cb_0, 0]
       ]) * -2

       H_horizontal = np.array([
           [J_aa_1x, J_ab_1x, J_ac_1x],
           [J_ba_1x, J_bb_1x, J_bc_1x],
           [J_ca_1x, J_cb_1x, J_cc_1x]
       ]) * -2 * np.cos(np.vdot(k,a_1))

       H_vertical = np.array([
           [J_aa_1y, J_ab_1y, J_ac_1y],
           [J_ba_1y, J_bb_1y, J_bc_1y],
           [J_ca_1y, J_cb_1y, J_cc_1y]
       ]) * -2 * np.cos(np.vdot(k,a_2))

       H_diagonal_p = np.array([
           [J_aa_2p, J_ab_2p, J_ac_2p],
           [J_ba_2p, J_bb_2p, J_bc_2p],
           [J_ca_2p, J_cb_2p, J_cc_2p]
       ]) * -2 * np.cos(np.vdot(k,a_1 + a_2))

       H_diagonal_m = np.array([
           [J_aa_2m, J_ab_2m, J_ac_2m],
           [J_ba_2m, J_bb_2m, J_bc_2m],
           [J_ca_2m, J_cb_2m, J_cc_2m]
       ]) * -2 * np.cos(np.vdot(k,a_1 - a_2))
       
       hamiltonian = (H_offsets + H_onsite + H_horizontal + H_vertical 
                      + H_diagonal_p + H_diagonal_m)
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
    d_ba = a*np.array([0.5, 0])
    d_ca = a*np.array([0.25, 0.25*3**0.5])
    d_cb = a*np.array([-0.25, 0.25*3**0.5])
    d_ab = -d_ba
    d_ac = -d_ca
    d_bc = -d_cb

    def A(time):
        drive = np.array([A_x*np.cos(omega*time), 
                                        -A_y*np.cos(omega*time + phi)])
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

def square_hamiltonian_driven(delta_a=0, 
                              delta_b=0, 
                              delta_c=0, 
                              J_ab_0=0, 
                              J_ac_0=0, 
                              J_bc_0=0, 
                              J_aa_1x=0, 
                              J_bb_1x=0, 
                              J_cc_1x=0, 
                              J_ab_1x=0, 
                              J_ac_1x=0, 
                              J_bc_1x=0,
                              J_aa_1y=0, 
                              J_bb_1y=0, 
                              J_cc_1y=0, 
                              J_ab_1y=0, 
                              J_ac_1y=0, 
                              J_bc_1y=0,
                              J_aa_2p=0, 
                              J_bb_2p=0, 
                              J_cc_2p=0, 
                              J_ab_2p=0, 
                              J_ac_2p=0, 
                              J_bc_2p=0,
                              J_aa_2m=0, 
                              J_bb_2m=0, 
                              J_cc_2m=0, 
                              J_ab_2m=0, 
                              J_ac_2m=0, 
                              J_bc_2m=0,  
                              A_x=0, 
                              A_y=0, 
                              omega=0, 
                              phi=0,
                              a=1):
    """Defines a bloch hamiltonian for the out-of-equillibrium square lattice.
    
    Parameters
    ----------
    delta_aa: float
        The on-site potential for the A orbital
    delta_bb: float
        The on-site potential for the B orbital
    delta_cc: float
        The on-site potential for the C orbital
    delta_ab: float
        The matrix element for the A and B orbitals at the same site
    delta_ac: float
        The matrix element for the A and C orbitals at the same site
    delta_bc: float
        The matrix element for the B and C orbitals at the same site
    J_aa_1x: float
        The tunneling parameter between nn A orbitals in the horizontal 
        direction
    J_bb_1x: float
        The tunneling parameter between nn B orbitals in the horizontal 
        direction
    J_cc_1x: float
        The tunneling parameter between nn C orbitals in the horizontal 
        direction
    J_ab_1x: float
        The tunneling parameter between nn A and B orbitals in the horizontal 
        direction
    J_ac_1x: float
        The tunneling parameter between nn A and C orbitals in the horizontal 
        direction
    J_bc_1x: float
        The tunneling parameter between nn B and C orbitals in the horizontal 
        direction
    J_aa_1y: float
        The tunneling parameter between nn A orbitals in the vertical direction
    J_bb_1y: float
        The tunneling parameter between nn B orbitals in the vertical direction
    J_cc_1y: float
        The tunneling parameter between nn C orbitals in the vertical direction
    J_ab_1y: float
        The tunneling parameter between nn A and B orbitals in the vertical 
        direction
    J_ac_1y: float
        The tunneling parameter between nn A and C orbitals in the vertical 
        direction
    J_bc_1y: float
        The tunneling parameter between nn B and C orbitals in the vertical 
        direction
    J_aa_2p: float
        The tunneling parameter between nn A orbitals in the diagonal a1+a2 
        direction
    J_bb_2p: float
        The tunneling parameter between nn B orbitals in the diagonal a1+a2 
        direction
    J_cc_2p: float
        The tunneling parameter between nn C orbitals in the diagonal a1+a2 
        direction
    J_ab_2p: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1+a2 direction
    J_ac_2p: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1+a2 direction
    J_bc_2p: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1+a2 direction
    J_aa_2m: float
        The tunneling parameter between nn A orbitals in the diagonal a1-a2 
        direction
    J_bb_2m: float
        The tunneling parameter between nn B orbitals in the diagonal a1-a2 
        direction
    J_cc_2m: float
        The tunneling parameter between nn C orbitals in the diagonal a1-a2 
        direction
    J_ab_2m: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1-a2 direction
    J_ac_2m: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1-a2 direction
    J_bc_2m: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1-a2 direction
    A_x: float
        The amplitude of the x component of the driving vector potential
    A_y: float
        The amplitude of the y component of the driving vector potential
    omega: float
        The angular frequency of the driving vector potential
    phi:
        The relative phase of the phi component of the vector potential
    a: float
        The lattice spacing

    Returns
    -------
    H: function
        The square bloch hamiltonian as a function of quasimomentum k and 
        time t"""
    a_1 = a*np.array([1,0])
    a_2 = a*np.array([0,1])

    J_ba_0 = np.conjugate(J_ab_0)
    J_ca_0 = np.conjugate(J_ac_0)
    J_cb_0 = np.conjugate(J_bc_0)
    J_ba_1x = np.conjugate(J_ab_1x)
    J_ca_1x = np.conjugate(J_ac_1x)
    J_cb_1x = np.conjugate(J_bc_1x)
    J_ba_1y = np.conjugate(J_ab_1y)
    J_ca_1y = np.conjugate(J_ac_1y)
    J_cb_1y = np.conjugate(J_bc_1y)
    J_ba_2p = np.conjugate(J_ab_2p)
    J_ca_2p = np.conjugate(J_ac_2p)
    J_cb_2p = np.conjugate(J_bc_2p)
    J_ba_2m = np.conjugate(J_ab_2m)
    J_ca_2m = np.conjugate(J_ac_2m)
    J_cb_2m = np.conjugate(J_bc_2m)

    def A(time):
        drive = np.array([A_x*np.cos(omega*time), 
                                        -A_y*np.cos(omega*time + phi)])
        return drive
    
    def H(k,t):
       H_offsets = np.array([
           [delta_a, 0, 0],
           [0, delta_b, 0],
           [0, 0, delta_c]
       ])

       H_onsite = np.array([
           [0, J_ab_0, J_ac_0],
           [J_ba_0, 0, J_bc_0],
           [J_ca_0, J_cb_0, 0]
       ]) * -2

       H_horizontal = np.array([
           [J_aa_1x, J_ab_1x, J_ac_1x],
           [J_ba_1x, J_bb_1x, J_bc_1x],
           [J_ca_1x, J_cb_1x, J_cc_1x]
       ]) * -2 * np.cos(np.vdot((k+A(t)),a_1))

       H_vertical = np.array([
           [J_aa_1y, J_ab_1y, J_ac_1y],
           [J_ba_1y, J_bb_1y, J_bc_1y],
           [J_ca_1y, J_cb_1y, J_cc_1y]
       ]) * -2 * np.cos(np.vdot((k+A(t)),a_2))

       H_diagonal_p = np.array([
           [J_aa_2p, J_ab_2p, J_ac_2p],
           [J_ba_2p, J_bb_2p, J_bc_2p],
           [J_ca_2p, J_cb_2p, J_cc_2p]
       ]) * -2 * np.cos(np.vdot((k+A(t)),a_1 + a_2))

       H_diagonal_m = np.array([
           [J_aa_2m, J_ab_2m, J_ac_2m],
           [J_ba_2m, J_bb_2m, J_bc_2m],
           [J_ca_2m, J_cb_2m, J_cc_2m]
       ]) * -2 * np.cos(np.vdot((k+A(t)),a_1 - a_2))
       
       hamiltonian = (H_offsets + H_onsite + H_horizontal + H_vertical 
                      + H_diagonal_p + H_diagonal_m)
       return hamiltonian
    
    return H

def square_hamiltonian_driven_finite_y(
                              L = 10,
                              delta_a=0, 
                              delta_b=0, 
                              delta_c=0, 
                              J_ab_0=0, 
                              J_ac_0=0, 
                              J_bc_0=0, 
                              J_aa_1x=0, 
                              J_bb_1x=0, 
                              J_cc_1x=0, 
                              J_ab_1x=0, 
                              J_ac_1x=0, 
                              J_bc_1x=0,
                              J_aa_1y=0, 
                              J_bb_1y=0, 
                              J_cc_1y=0, 
                              J_ab_1y=0, 
                              J_ac_1y=0, 
                              J_bc_1y=0,
                              J_aa_2p=0, 
                              J_bb_2p=0, 
                              J_cc_2p=0, 
                              J_ab_2p=0, 
                              J_ac_2p=0, 
                              J_bc_2p=0,
                              J_aa_2m=0, 
                              J_bb_2m=0, 
                              J_cc_2m=0, 
                              J_ab_2m=0, 
                              J_ac_2m=0, 
                              J_bc_2m=0,  
                              A_x=0, 
                              A_y=0, 
                              omega=0, 
                              phi=0,
                              a=1):
    """Defines a bloch hamiltonian for the out-of-equillibrium square lattice,
    cut along the x direction.
    
    Parameters
    ----------
    L: int
        The number of layres in the y direction to use
    delta_aa: float
        The on-site potential for the A orbital
    delta_bb: float
        The on-site potential for the B orbital
    delta_cc: float
        The on-site potential for the C orbital
    delta_ab: float
        The matrix element for the A and B orbitals at the same site
    delta_ac: float
        The matrix element for the A and C orbitals at the same site
    delta_bc: float
        The matrix element for the B and C orbitals at the same site
    J_aa_1x: float
        The tunneling parameter between nn A orbitals in the horizontal 
        direction
    J_bb_1x: float
        The tunneling parameter between nn B orbitals in the horizontal 
        direction
    J_cc_1x: float
        The tunneling parameter between nn C orbitals in the horizontal 
        direction
    J_ab_1x: float
        The tunneling parameter between nn A and B orbitals in the horizontal 
        direction
    J_ac_1x: float
        The tunneling parameter between nn A and C orbitals in the horizontal 
        direction
    J_bc_1x: float
        The tunneling parameter between nn B and C orbitals in the horizontal 
        direction
    J_aa_1y: float
        The tunneling parameter between nn A orbitals in the vertical direction
    J_bb_1y: float
        The tunneling parameter between nn B orbitals in the vertical direction
    J_cc_1y: float
        The tunneling parameter between nn C orbitals in the vertical direction
    J_ab_1y: float
        The tunneling parameter between nn A and B orbitals in the vertical 
        direction
    J_ac_1y: float
        The tunneling parameter between nn A and C orbitals in the vertical 
        direction
    J_bc_1y: float
        The tunneling parameter between nn B and C orbitals in the vertical 
        direction
    J_aa_2p: float
        The tunneling parameter between nn A orbitals in the diagonal a1+a2 
        direction
    J_bb_2p: float
        The tunneling parameter between nn B orbitals in the diagonal a1+a2 
        direction
    J_cc_2p: float
        The tunneling parameter between nn C orbitals in the diagonal a1+a2 
        direction
    J_ab_2p: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1+a2 direction
    J_ac_2p: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1+a2 direction
    J_bc_2p: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1+a2 direction
    J_aa_2m: float
        The     tunneling parameter between nn A orbitals in the diagonal a1-a2 
        direction
    J_bb_2m: float
        The tunneling parameter between nn B orbitals in the diagonal a1-a2 
        direction
    J_cc_2m: float
        The tunneling parameter between nn C orbitals in the diagonal a1-a2 
        direction
    J_ab_2m: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1-a2 direction
    J_ac_2m: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1-a2 direction
    J_bc_2m: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1-a2 direction
    A_x: float
        The amplitude of the x component of the driving vector potential
    A_y: float
        The amplitude of the y component of the driving vector potential
    omega: float
        The angular frequency of the driving vector potential
    phi:
        The relative phase of the phi component of the vector potential
    a: float
        The lattice spacing

    Returns
    -------
    H: function
        The square bloch hamiltonian as a function of quasimomentum k and 
        time t"""
    a_1 = a*np.array([1,0])
    a_2 = a*np.array([0,1])

    J_ba_0 = np.conjugate(J_ab_0)
    J_ca_0 = np.conjugate(J_ac_0)
    J_cb_0 = np.conjugate(J_bc_0)
    J_ba_1x = np.conjugate(J_ab_1x)
    J_ca_1x = np.conjugate(J_ac_1x)
    J_cb_1x = np.conjugate(J_bc_1x)
    J_ba_1y = np.conjugate(J_ab_1y)
    J_ca_1y = np.conjugate(J_ac_1y)
    J_cb_1y = np.conjugate(J_bc_1y)
    J_ba_2p = np.conjugate(J_ab_2p)
    J_ca_2p = np.conjugate(J_ac_2p)
    J_cb_2p = np.conjugate(J_bc_2p)
    J_ba_2m = np.conjugate(J_ab_2m)
    J_ca_2m = np.conjugate(J_ac_2m)
    J_cb_2m = np.conjugate(J_bc_2m)

    
    def A(time):
        drive = np.array([A_x*np.cos(omega*time), 
                                        -A_y*np.cos(omega*time + phi)])
        return drive
    
    def H(k,t):
        hamiltonian = np.zeros((3*L,3*L), dtype='complex')

        on_site = np.array([
            [delta_a, -2*J_ab_0, -2*J_ac_0],
            [-2*J_ba_0, delta_b, -2*J_bc_0],
            [-2*J_ca_0, -2*J_cb_0, delta_c]
        ])

        horizontal = np.array([
            [J_aa_1x, J_ab_1x, J_ac_1x],
            [J_ba_1x, J_bb_1x, J_bc_1x],
            [J_ca_1x, J_cb_1x, J_cc_1x]
        ]) * -2*np.cos((k+A(t)[0])*a)

        vertical_pos = np.array([
            [J_aa_1y, J_ab_1y, J_ac_1y],
            [J_ba_1y, J_bb_1y, J_bc_1y],
            [J_ca_1y, J_cb_1y, J_cc_1y]
        ])*-np.exp(1j*A(t)[1]*a)

        vertical_neg = np.array([
            [J_aa_1y, J_ab_1y, J_ac_1y],
            [J_ba_1y, J_bb_1y, J_bc_1y],
            [J_ca_1y, J_cb_1y, J_cc_1y]
        ])*-np.exp(-1j*A(t)[1]*a)

        diagonal_p_pos = np.array([
            [J_aa_2p, J_ab_2p, J_ac_2p],
            [J_ba_2p, J_bb_2p, J_bc_2p],
            [J_ca_2p, J_cb_2p, J_cc_2p]
        ])*-np.exp(1j*(k+A(t)[0]+A(t)[1])*a)

        diagonal_p_neg = np.array([
            [J_aa_2p, J_ab_2p, J_ac_2p],
            [J_ba_2p, J_bb_2p, J_bc_2p],
            [J_ca_2p, J_cb_2p, J_cc_2p]
        ])*-np.exp(-1j*(k+A(t)[0]+A(t)[1])*a)

        diagonal_m_pos = np.array([
            [J_aa_2m, J_ab_2m, J_ac_2m],
            [J_ba_2m, J_bb_2m, J_bc_2m],
            [J_ca_2m, J_cb_2m, J_cc_2m]
        ])*-np.exp(1j*(-k-A(t)[0]+A(t)[1])*a)

        diagonal_m_neg = np.array([
            [J_aa_2m, J_ab_2m, J_ac_2m],
            [J_ba_2m, J_bb_2m, J_bc_2m],
            [J_ca_2m, J_cb_2m, J_cc_2m]
        ])*-np.exp(-1j*(-k-A(t)[0]+A(t)[1])*a)


        for i in range(L):
            for j in range(L):
                if i == j:
                    hamiltonian[3*i:3*i+3,3*j:3*j+3] = on_site + horizontal
                elif i+1 == j:
                    hamiltonian[3*i:3*i+3,3*j:3*j+3] = (vertical_pos 
                                                        + diagonal_p_pos 
                                                        + diagonal_m_pos)
                elif i-1 == j:
                    hamiltonian[3*i:3*i+3,3*j:3*j+3] = (vertical_neg 
                                                        + diagonal_p_neg
                                                        + diagonal_m_neg)
                                            
        return hamiltonian
    
    return H

def square_hamiltonian_driven_finite_x(
                              L = 10,
                              delta_a=0, 
                              delta_b=0, 
                              delta_c=0, 
                              J_ab_0=0, 
                              J_ac_0=0, 
                              J_bc_0=0, 
                              J_aa_1x=0, 
                              J_bb_1x=0, 
                              J_cc_1x=0, 
                              J_ab_1x=0, 
                              J_ac_1x=0, 
                              J_bc_1x=0,
                              J_aa_1y=0, 
                              J_bb_1y=0, 
                              J_cc_1y=0, 
                              J_ab_1y=0, 
                              J_ac_1y=0, 
                              J_bc_1y=0,
                              J_aa_2p=0, 
                              J_bb_2p=0, 
                              J_cc_2p=0, 
                              J_ab_2p=0, 
                              J_ac_2p=0, 
                              J_bc_2p=0,
                              J_aa_2m=0, 
                              J_bb_2m=0, 
                              J_cc_2m=0, 
                              J_ab_2m=0, 
                              J_ac_2m=0, 
                              J_bc_2m=0,  
                              A_x=0, 
                              A_y=0, 
                              omega=0, 
                              phi=0,
                              a=1):
    """Defines a bloch hamiltonian for the out-of-equillibrium square lattice,
    cut along the y direction.
    
    Parameters
    ----------
    L: int
        The number of layers in the x direction to use
    delta_aa: float
        The on-site potential for the A orbital
    delta_bb: float
        The on-site potential for the B orbital
    delta_cc: float
        The on-site potential for the C orbital
    delta_ab: float
        The matrix element for the A and B orbitals at the same site
    delta_ac: float
        The matrix element for the A and C orbitals at the same site
    delta_bc: float
        The matrix element for the B and C orbitals at the same site
    J_aa_1x: float
        The tunneling parameter between nn A orbitals in the horizontal 
        direction
    J_bb_1x: float
        The tunneling parameter between nn B orbitals in the horizontal 
        direction
    J_cc_1x: float
        The tunneling parameter between nn C orbitals in the horizontal 
        direction
    J_ab_1x: float
        The tunneling parameter between nn A and B orbitals in the horizontal 
        direction
    J_ac_1x: float
        The tunneling parameter between nn A and C orbitals in the horizontal 
        direction
    J_bc_1x: float
        The tunneling parameter between nn B and C orbitals in the horizontal 
        direction
    J_aa_1y: float
        The tunneling parameter between nn A orbitals in the vertical direction
    J_bb_1y: float
        The tunneling parameter between nn B orbitals in the vertical direction
    J_cc_1y: float
        The tunneling parameter between nn C orbitals in the vertical direction
    J_ab_1y: float
        The tunneling parameter between nn A and B orbitals in the vertical 
        direction
    J_ac_1y: float
        The tunneling parameter between nn A and C orbitals in the vertical 
        direction
    J_bc_1y: float
        The tunneling parameter between nn B and C orbitals in the vertical 
        direction
    J_aa_2p: float
        The tunneling parameter between nn A orbitals in the diagonal a1+a2 
        direction
    J_bb_2p: float
        The tunneling parameter between nn B orbitals in the diagonal a1+a2 
        direction
    J_cc_2p: float
        The tunneling parameter between nn C orbitals in the diagonal a1+a2 
        direction
    J_ab_2p: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1+a2 direction
    J_ac_2p: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1+a2 direction
    J_bc_2p: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1+a2 direction
    J_aa_2m: float
        The     tunneling parameter between nn A orbitals in the diagonal a1-a2 
        direction
    J_bb_2m: float
        The tunneling parameter between nn B orbitals in the diagonal a1-a2 
        direction
    J_cc_2m: float
        The tunneling parameter between nn C orbitals in the diagonal a1-a2 
        direction
    J_ab_2m: float
        The tunneling parameter between nn A and B orbitals in the diagonal 
        a1-a2 direction
    J_ac_2m: float
        The tunneling parameter between nn A and C orbitals in the diagonal 
        a1-a2 direction
    J_bc_2m: float
        The tunneling parameter between nn B and C orbitals in the diagonal 
        a1-a2 direction
    A_x: float
        The amplitude of the x component of the driving vector potential
    A_y: float
        The amplitude of the y component of the driving vector potential
    omega: float
        The angular frequency of the driving vector potential
    phi:
        The relative phase of the phi component of the vector potential
    a: float
        The lattice spacing

    Returns
    -------
    H: function
        The square bloch hamiltonian as a function of quasimomentum k and 
        time t"""
    a_1 = a*np.array([1,0])
    a_2 = a*np.array([0,1])

    J_ba_0 = np.conjugate(J_ab_0)
    J_ca_0 = np.conjugate(J_ac_0)
    J_cb_0 = np.conjugate(J_bc_0)
    J_ba_1x = np.conjugate(J_ab_1x)
    J_ca_1x = np.conjugate(J_ac_1x)
    J_cb_1x = np.conjugate(J_bc_1x)
    J_ba_1y = np.conjugate(J_ab_1y)
    J_ca_1y = np.conjugate(J_ac_1y)
    J_cb_1y = np.conjugate(J_bc_1y)
    J_ba_2p = np.conjugate(J_ab_2p)
    J_ca_2p = np.conjugate(J_ac_2p)
    J_cb_2p = np.conjugate(J_bc_2p)
    J_ba_2m = np.conjugate(J_ab_2m)
    J_ca_2m = np.conjugate(J_ac_2m)
    J_cb_2m = np.conjugate(J_bc_2m)

    
    def A(time):
        drive = np.array([A_x*np.cos(omega*time), 
                                        -A_y*np.cos(omega*time + phi)])
        return drive
    
    def H(k,t):
        hamiltonian = np.zeros((3*L,3*L), dtype='complex')

        on_site = np.array([
            [delta_a, -2*J_ab_0, -2*J_ac_0],
            [-2*J_ba_0, delta_b, -2*J_bc_0],
            [-2*J_ca_0, -2*J_cb_0, delta_c]
        ])

        vertical = np.array([
            [J_aa_1y, J_ab_1y, J_ac_1y],
            [J_ba_1y, J_bb_1y, J_bc_1y],
            [J_ca_1y, J_cb_1y, J_cc_1y]
        ]) * -2*np.cos((k+A(t)[1])*a)

        horizontal_pos = np.array([
            [J_aa_1x, J_ab_1x, J_ac_1x],
            [J_ba_1x, J_bb_1x, J_bc_1x],
            [J_ca_1x, J_cb_1x, J_cc_1x]
        ])*-np.exp(1j*A(t)[0]*a)

        horizontal_neg = np.array([
            [J_aa_1x, J_ab_1x, J_ac_1x],
            [J_ba_1x, J_bb_1x, J_bc_1x],
            [J_ca_1x, J_cb_1x, J_cc_1x]
        ])*-np.exp(-1j*A(t)[0]*a)

        diagonal_p_pos = np.array([
            [J_aa_2p, J_ab_2p, J_ac_2p],
            [J_ba_2p, J_bb_2p, J_bc_2p],
            [J_ca_2p, J_cb_2p, J_cc_2p]
        ])*-np.exp(1j*(k+A(t)[1]+A(t)[0])*a)

        diagonal_p_neg = np.array([
            [J_aa_2p, J_ab_2p, J_ac_2p],
            [J_ba_2p, J_bb_2p, J_bc_2p],
            [J_ca_2p, J_cb_2p, J_cc_2p]
        ])*-np.exp(-1j*(k+A(t)[1]+A(t)[0])*a)

        diagonal_m_pos = np.array([
            [J_aa_2m, J_ab_2m, J_ac_2m],
            [J_ba_2m, J_bb_2m, J_bc_2m],
            [J_ca_2m, J_cb_2m, J_cc_2m]
        ])*-np.exp(1j*(-k-A(t)[1]+A(t)[0])*a)

        diagonal_m_neg = np.array([
            [J_aa_2m, J_ab_2m, J_ac_2m],
            [J_ba_2m, J_bb_2m, J_bc_2m],
            [J_ca_2m, J_cb_2m, J_cc_2m]
        ])*-np.exp(-1j*(-k-A(t)[1]+A(t)[0])*a)


        for i in range(L):
            for j in range(L):
                if i == j:
                    hamiltonian[3*i:3*i+3,3*j:3*j+3] = on_site + vertical
                elif i+1 == j:
                    hamiltonian[3*i:3*i+3,3*j:3*j+3] = (horizontal_pos 
                                                        + diagonal_p_pos 
                                                        + diagonal_m_pos)
                elif i-1 == j:
                    hamiltonian[3*i:3*i+3,3*j:3*j+3] = (horizontal_neg 
                                                        + diagonal_p_neg
                                                        + diagonal_m_neg)
                                            
        return hamiltonian
    
    return H

def square_hamiltonian_driven_compact(J,
                              n1,
                              n2,
                              m,
                              num_points,
                              num_steps):
    """Finds the bloch hamiltonian corresponding to given hoppings and drive.
    
    Parameters
    ----------
    J: function
        The hoppings such that J[i,j,m] is the hopping matrix for the direction
        n1[i,j]*a1 + n2[i,j]*a2 and fourier component exp(imwt)
    n1: 2D array
    n2: 2D array
    m: 1D array

    Returns
    -------
    H: 5D array
        The bloch hamiltonian where H[t,i,j] is the bloch hamiltonian at time
        T / num_steps * t and position i / num_points * b1 + j / num_points * b2
    """
    hamiltonian = np.zeros((num_steps,num_points,num_points,3,3), dtype='complex')
    t = np.linspace(0,1,num_steps)
    k1 = np.linspace(0,1,num_points)
    k2 = np.linspace(0,1,num_points)

    t = t[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
    k1 = k1[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    k2 = k2[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]

    t = np.repeat(t,num_points,1)
    t = np.repeat(t,num_points,2)
    t = np.repeat(t,3,3)
    t = np.repeat(t,3,4)

    k1 = np.repeat(k1,num_steps,0)
    k1 = np.repeat(k1,num_points,2)
    k1 = np.repeat(k1,3,3)
    k1 = np.repeat(k1,3,4)

    k2 = np.repeat(k2,num_steps,0)
    k2 = np.repeat(k2,num_points,1)
    k2 = np.repeat(k2,3,3)
    k2 = np.repeat(k2,3,4)

    for k in range(len(n1)):
        for l in range(len(n2)):
            for p in range(len(m)):
                J = J[k,l,p][np.newaxis,np.newaxis,np.newaxis,:,:]
                J = np.repeat(J,num_steps,0)
                J = np.repeat(J,num_points,1)
                J = np.repeat(J,num_points,2)
                exponent = np.exp(1j*2*np.pi*(m[p]*t+n1[k]*k1+n2[l]*k2))
                hamiltonian += J * exponent
    return hamiltonian

def square_hamiltonian_finite_geometry_compact(J,
                                                n1,
                                                n2,
                                                L=10,
                                                omega=0, 
                                                phi=0,
                                                a=1):
    """Returns two hamiltonians for finite geometries on for each cut direction.
    
    Parameters
    ----------
    J: function
        The hoppings such that J(t)[i,j] is the hopping matrix for the direction
        n1[i,j]*a1 + n2[i,j]*a2
    n1: 2D array
    n2: 2D array
    L: int
        The number of layers to consider.
    omega: float
        The angular frequency of the driving vector potential
    phi:
        The relative phase of the phi component of the vector potential
    a: float
        The lattice spacing

    Returns
    -------
    H_1: function
        The bloch hamiltonian for the cut along the a1 direction
    H_2: function
        The bloch hamiltonian for the cut along the a2 direction
    """
    a1 = a*np.array([1,0])
    a2 = a*np.array([0,1])


    # Cut along a1
    def H_1(k,t):
        hamiltonian = np.zeros((3*L,3*L), dtype='complex')
        for m2 in range(L):
            for j in range(n2.shape[1]):
                if m2 + n2[0,j] >= 0 and m2 + n2[0,j] < L:
                    for i in range(n1.shape[0]):
                        hamiltonian[3*(m2+n2[0,j]):3*(m2+n2[0,j])+3,
                                    3*m2:3*m2+3] += J(t)[i,j]*np.exp(
                            1j*(n1[i,j]*k*np.linalg.norm(a1)))
        return hamiltonian
    
    # Cut along a2
    def H_2(k,t):
        hamiltonian = np.zeros((3*L,3*L), dtype='complex')
        for m1 in range(L):
            for i in range(n1.shape[1]):
                if m1 + n1[i,0] >= 0 and m1 + n1[i,0] < L:
                    for j in range(n2.shape[0]):
                        hamiltonian[3*(m1+n1[i,0]):3*(m1+n1[i,0])+3,
                                    3*m1:3*m1+3] += J(t)[i,j]*np.exp(
                            1j*(n2[i,j]*k*np.linalg.norm(a2)))
        return hamiltonian
    
    return H_1, H_2