import numpy as np
import os
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D, locate_nodes
from tight_binding.topology import compute_zak_phase, locate_dirac_strings

plotting = True
slicing = False
nodes = True
locate_ds = False


# SAVING
main_directory = 'figures/square/SP3/driven/bandstructures'

# LOCATE DIRAC STRINGS
directions = np.array([
    [1,0],
    [0,1],
    [1,1],
    [1,-1]
])

perpendicular_directions = np.array([
    [0,1],
    [1,0],
    [1,-1],
    [1,1]
])

# PARAMETERS
delta_A_scan = np.array([-11,-9,-7,-5,-3])
delta_C_scan = np.array([-3,-1,1,3,5])
omega = 10
A_x = 1

## OTHER
a_1 = np.array([1,0])
a_2 = np.array([0,1])
num_points = 100
num_steps = 100
num_lines = 100
lowest_quasi_energy = -np.pi / 4
offsets = np.zeros((3,2))


for i in range(len(delta_A_scan)):
    for j in range(len(delta_C_scan)):
        delta_A = delta_A_scan[i]
        delta_C = delta_C_scan[j]
        delta_B = - delta_A - delta_C

        name = 'SP3_driven_Ax_w_dA_dC_{A_x}_{omega}_{delta_A}_{delta_C}'.format(
            A_x=A_x, omega=omega, delta_A=delta_A, delta_C=delta_C
        )
        
        directory = main_directory + '/' + name
        if not os.path.isdir(directory):
            os.mkdir(directory)

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
            
            grid_dir = directory + '/' + name + '_grids'
            if not os.path.isdir(grid_dir):
                os.mkdir(grid_dir)
            np.save(grid_dir + '/' + name + '_grid', energies)

            save = directory + '/' + name + '.png'
            plot_bandstructure2D(energies, a_1, a_2, save, 
                                bands_to_plot = [1,1,1], 
                                lowest_quasi_energy=lowest_quasi_energy, 
                                discontinuity_threshold = 0.05, show_plot=False)
            
            save = directory + '/' + name + '_band0' + '.png'
            plot_bandstructure2D(energies, a_1, a_2, save, 
                                bands_to_plot = [1,0,0], 
                                lowest_quasi_energy=lowest_quasi_energy, 
                                discontinuity_threshold = 0.05, show_plot=False)
            
            save = directory + '/' + name + '_band1' + '.png'
            plot_bandstructure2D(energies, a_1, a_2, save, 
                                bands_to_plot = [0,1,0], 
                                lowest_quasi_energy=lowest_quasi_energy, 
                                discontinuity_threshold = 0.05, show_plot=False)
            
            save = directory + '/' + name + '_band2' + '.png'
            plot_bandstructure2D(energies, a_1, a_2, save, 
                                bands_to_plot = [0,0,1], 
                                lowest_quasi_energy=lowest_quasi_energy, 
                                discontinuity_threshold = 0.05, show_plot=False)

        # LOCATING NODES
        if nodes:
            node_dir = directory + '/' + name + '_nodes'
            if not os.path.isdir(node_dir):
                os.mkdir(node_dir)
            title = '$\Delta_A={delta_A}, \Delta_C={delta_C}$'.format(
                delta_A=delta_A, delta_C=delta_C
            )
            locate_nodes(energies, a_1, a_2, node_dir + '/' + name + '_nodes',
                         show_plot=False, title=title)
            


        # LOCATING DIRAC STRINGS
        if locate_ds:
            ds_dir = directory + '/' + name + '_ds'
            if not os.path.isdir(ds_dir):
                os.mkdir(ds_dir)

            for k in range(len(directions)):
                direction = directions[k]
                perpendicular_direction = perpendicular_directions[k]
                save = (ds_dir + '/' + name + '_ds' 
                    + '_{x}_{y}_{xx}_{yy}'.format(
                    x=direction[0], y=direction[1], 
                    xx=perpendicular_direction[0], 
                    yy=perpendicular_direction[1])) + '.png'
                locate_dirac_strings(H, direction, perpendicular_direction, num_lines, 
                                    save, a_1, a_2, offsets, num_points, omega, 
                                    num_steps, lowest_quasi_energy, 
                                    show_plot=False)

        # SLICING

        if slicing:
            slice_dir = directory + '/' + name + '_sliced'
            if not os.path.isdir(slice_dir):
                os.mkdir(slice_dir)
            ## X slices
            xslice_dir = slice_dir + '/xslices'
            if not os.path.isdir(xslice_dir):
                os.mkdir(xslice_dir)
            for k in range(100):
                save = xslice_dir + '/' + name + '_xslice{k}'.format(k=k)  + '.png'
                start = np.array([k / 100, 0])
                end = np.array([k / 100, 1])
                kx = k / 100 * 2
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
                plt.savefig(save)
                plt.close()

            # Y slices
            yslice_dir = slice_dir + '/yslices'
            if not os.path.isdir(yslice_dir):
                os.mkdir(yslice_dir)
            for k in range(100):
                save = yslice_dir + '/' + name + '_yslice{k}'.format(k=k)  + '.png'
                start = np.array([0, k / 100])
                end = np.array([1, k / 100])
                ky = k / 100 * 2
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
                plt.savefig(save)
                plt.close()