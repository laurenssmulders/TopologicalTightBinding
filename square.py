import numpy as np
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D
from tight_binding.topology import compute_zak_phase, locate_dirac_strings

# PARAMETERS
## TO VARY
A_x = 1.5
J_2 = 0.8
delta_C = 0


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
    delta_aa=0,
    delta_bb=0,
    delta_cc=0,
    delta_ab=1,
    delta_ac=1,
    delta_bc=1,
    J_bc_1x=1,
    J_ac_1y=1
)

# BANDSTRUCTURE
energies, blochvectors = compute_bandstructure2D(H, a_1, a_2, num_points, omega,
                                                 num_steps, lowest_quasi_energy)

plot_bandstructure2D(energies, a_1, a_2, 'test.png', bands_to_plot = [1,1,1], 
                     lowest_quasi_energy=lowest_quasi_energy, 
                     discontinuity_threshold = 0.05)

# ZAK PHASES

#for i in range(100):
#    start = np.array([0,i / 100])
#    end = np.array([1,i / 100])
#    k = i / 100 * 2

#    offsets = np.zeros((3,2))
#    zak_phase, energies = compute_zak_phase(H, a_1, a_2, offsets, start, end, num_points, 
#                                omega, num_steps, lowest_quasi_energy)
#    zak_phase = np.rint(np.real(zak_phase / np.pi))

#    fig = plt.figure()
#    plt.plot(energies[:,0], label='band 0')
#    plt.plot(energies[:,1], label='band 1')
#    plt.plot(energies[:,2], label='band 2')
#    plt.legend()
#    plt.title('$k_y  = {k}\pi$ Zak Phases: {zak_phase}'.format(k=k, zak_phase=zak_phase))
#    plt.savefig('figures/square/square_Ax_J2_dC_1p5_0p8_0p0_yslices/square_Ax_J2_dC_1p5_0p0_0p5_yslice{i}'.format(i=i))
#    plt.close()

# DIRAC STRINGS

#offsets = np.zeros((3,3))
#direction = np.array([1,0])
#perpendicular_direction = np.array([0,1])
#num_lines = 100

#locate_dirac_strings(H, direction, perpendicular_direction, num_lines, 
#                     'test.png', a_1, a_2, offsets, num_points, omega, 
#                     num_steps, lowest_quasi_energy)