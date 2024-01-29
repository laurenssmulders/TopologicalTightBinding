import numpy as np
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.topology import compute_zak_phase, locate_dirac_strings

# PARAMETERS
## TO VARY
A_x = 1
J_2 = 1
delta_C = 2


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

plot_bandstructure2D(energies, a_1, a_2, 'test.png', bands_to_plot = [1,1,1], 
                     lowest_quasi_energy=lowest_quasi_energy, 
                     discontinuity_threshold = 0.05)

# ZAK PHASES
#start = np.array([0,0])
#end = np.array([1,0])

#offsets = np.zeros((3,3))
#zak_phase, energies = compute_zak_phase(H, a_1, a_2, offsets, start, end, num_points, 
#                              omega, num_steps, lowest_quasi_energy)
#zak_phase = np.rint(np.real(zak_phase / np.pi))

#print(zak_phase)

#fig = plt.figure()
#plt.plot(energies[:,0], label='band 0')
#plt.plot(energies[:,1], label='band 1')
#plt.plot(energies[:,2], label='band 2')
#plt.legend()
#plt.show()

# DIRAC STRINGS

#offsets = np.zeros((3,3))
#direction = np.array([1,0])
#perpendicular_direction = np.array([0,1])
#num_lines = 100

#locate_dirac_strings(H, direction, perpendicular_direction, num_lines, 
#                     'test.png', a_1, a_2, offsets, num_points, omega, 
#                     num_steps, lowest_quasi_energy)