import numpy as np
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D, locate_nodes
from tight_binding.topology import compute_zak_phase, locate_dirac_strings

plotting = False
slicing = False
zak = True
locate_ds = False

# PARAMETERS
delta_A = 5
delta_C = -5
omega = 10
A_x = 1

delta_B = -delta_A - delta_C

## OTHER
a_1 = np.array([1,0])
a_2 = np.array([0,1])
num_points = 100
num_steps = 100
num_lines = 100
lowest_quasi_energy = -np.pi / 4
offsets = np.zeros((3,2))

# BLOCH HAMILTONIAN
H = square_hamiltonian_driven(
    delta_a=delta_A,
    delta_b=delta_B,
    delta_c=delta_C,
    J_ab_0=1,
    J_ac_0=1,
    J_bc_0=1,
    J_ac_1x=1,
    J_bc_1y=1,
    J_ab_2m=1,
    A_x=A_x,
    omega=omega
)

# BANDSTRUCTURE
if plotting:
    energies, blochvectors = compute_bandstructure2D(H, a_1, a_2, num_points, 
                                                     omega, num_steps, 
                                                     lowest_quasi_energy)


    plot_bandstructure2D(energies, a_1, a_2, 'test.png', 
                         bands_to_plot = [1,1,1], 
                         lowest_quasi_energy=lowest_quasi_energy, 
                         discontinuity_threshold = 0.05)

# ZAK PHASE
if zak:
    start = np.array([0,0])
    end = np.array([1,1])

    zak_phase, energies = compute_zak_phase(H, a_1, a_2, offsets, start, end, 
                                            num_points, omega, num_steps, 
                                            lowest_quasi_energy)
    zak_phase = np.rint(np.real(zak_phase / np.pi))
    print(zak_phase)

# LOCATING DIRAC STRINGS
if locate_ds:
    direction = np.array([1,0])
    perpendicular_direction = np.array([0,1])
    locate_dirac_strings(H, direction, perpendicular_direction, num_lines, 
                         'test.png', a_1, a_2, offsets, num_points, omega, 
                         num_steps, lowest_quasi_energy)

# SLICING

if slicing:
    ## X slices
    for i in range(100):
        start = np.array([i / 100, 0])
        end = np.array([i / 100, 1])
        kx = i / 100 * 2
        ky = np.linspace(0,1,100) * 2*np.pi

        zak_phase, energies = compute_zak_phase(H, a_1, a_2, offsets, start, 
                                                end, num_points, omega, 
                                                num_steps, lowest_quasi_energy)
        zak_phase = np.rint(np.real(zak_phase / np.pi))


        fig = plt.figure()
        plt.plot(ky,energies[:,0], label='band 0')
        plt.plot(ky,energies[:,1], label='band 1')
        plt.plot(ky,energies[:,2], label='band 2')
        plt.legend()
        plt.title('$k_x  = {kx}\pi$ Zak Phases: {zak_phase}'.format(kx=kx, 
                                                        zak_phase=zak_phase))
        plt.xlabel('$k_y$')
        plt.xticks([0,np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                   [0,'$\pi / 2$', '$\pi$', '$3\pi / 2$', '$2\pi$'])
        plt.ylabel('$E / J$')
        plt.savefig('figures/square/SP3/driven/bandstructures/scan1/SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}/SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}_sliced/xslices/SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}_xslice{i}'.format(A_x=A_x, omega=omega, delta_A=delta_A, delta_C=delta_C, i=i))
        plt.close()

    # Y slices
    for i in range(100):
        start = np.array([0, i / 100])
        end = np.array([1, i / 100])
        ky = i / 100 * 2
        kx = np.linspace(0,1,100) * 2*np.pi

        offsets = np.zeros((3,2))
        zak_phase, energies = compute_zak_phase(H, a_1, a_2, offsets, start, 
                                                end, num_points, omega, 
                                                num_steps, lowest_quasi_energy)
        zak_phase = np.rint(np.real(zak_phase / np.pi))


        fig = plt.figure()
        plt.plot(kx,energies[:,0], label='band 0')
        plt.plot(kx,energies[:,1], label='band 1')
        plt.plot(kx,energies[:,2], label='band 2')
        plt.legend()
        plt.title('$k_y  = {ky}\pi$ Zak Phases: {zak_phase}'.format(ky=ky, zak_phase=zak_phase))
        plt.xlabel('$k_x$')
        plt.xticks([0,np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                   [0,'$\pi / 2$', '$\pi$', '$3\pi / 2$', '$2\pi$'])
        plt.ylabel('$E / J$')
        plt.savefig('figures/square/SP3/driven/bandstructures/SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}/SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}_sliced/yslices/SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}_yslice{i}'.format(A_x=A_x, omega=omega, delta_A=delta_A, delta_C=delta_C, i=i))
        plt.close()