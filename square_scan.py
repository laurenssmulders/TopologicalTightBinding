import numpy as np
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.topology import compute_zak_phase, locate_dirac_strings

scan = np.linspace(-5,5,21)

for i in range(len(scan)):
    print(i)
    # PARAMETERS
    ## TO VARY
    A_x = 1.5
    J_2 = 0.8
    delta_C = scan[i]


    ## DEPENDENT
    delta_A = - delta_C
    delta_B = - delta_A - delta_C
    J_1 = 1
    omega = 20

    ## OTHER
    a_1 = np.array([1,0])
    a_2 = np.array([0,1])
    num_points = 100
    num_steps = 100
    lowest_quasi_energy = -np.pi


    # BLOCH HAMILTONIAN
    H = square_hamiltonian_driven(
        delta_aa=delta_A,
        delta_bb=delta_B,
        delta_cc=delta_C,
        J_bb_1x=J_1,
        J_bc_1x=J_1,
        J_cc_1x=J_1,
        J_aa_1y=J_1,
        J_ac_1y=J_1,
        J_cc_1y=J_1,
        J_aa_2=J_2,
        J_ab_2=J_2,
        J_ac_2=J_2,
        J_bb_2=J_2,
        J_bc_2=J_2,
        J_cc_2=J_2,
        A_x=A_x,
        omega=omega
    )

    # BANDSTRUCTURE
    energies, blochvectors = compute_bandstructure2D(H, a_1, a_2, num_points, omega,
                                                    num_steps, lowest_quasi_energy)

    plot_bandstructure2D(energies, a_1, a_2, 'figures/square/square_Ax_J2_dC_{A_x}_{J_2}_{delta_C}_all.png'.format(A_x=A_x, J_2=J_2, delta_C=delta_C), 
                        bands_to_plot = [1,1,1], lowest_quasi_energy=lowest_quasi_energy, 
                        discontinuity_threshold = 0.08)

    plot_bandstructure2D(energies, a_1, a_2, 'figures/square/square_Ax_J2_dC_{A_x}_{J_2}_{delta_C}_0.png'.format(A_x=A_x, J_2=J_2, delta_C=delta_C), 
                        bands_to_plot = [1,0,0], lowest_quasi_energy=lowest_quasi_energy, 
                        discontinuity_threshold = 0.08)

    plot_bandstructure2D(energies, a_1, a_2, 'figures/square/square_Ax_J2_dC_{A_x}_{J_2}_{delta_C}_1.png'.format(A_x=A_x, J_2=J_2, delta_C=delta_C), 
                        bands_to_plot = [0,1,0], lowest_quasi_energy=lowest_quasi_energy, 
                        discontinuity_threshold = 0.08)

    plot_bandstructure2D(energies, a_1, a_2, 'figures/square/square_Ax_J2_dC_{A_x}_{J_2}_{delta_C}_2.png'.format(A_x=A_x, J_2=J_2, delta_C=delta_C), 
                        bands_to_plot = [0,0,1], lowest_quasi_energy=lowest_quasi_energy, 
                        discontinuity_threshold = 0.08)