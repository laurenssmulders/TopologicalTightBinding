import numpy as np
import os
import matplotlib.pyplot as plt
from tight_binding.hamiltonians import square_hamiltonian_driven, square_hamiltonian_driven_finite_x, square_hamiltonian_driven_finite_y
from tight_binding.bandstructure import compute_bandstructure2D, plot_bandstructure2D, locate_nodes, sort_energy_path
from tight_binding.topology import compute_zak_phase, compute_patch_euler_class
from tight_binding.diagonalise import compute_eigenstates

plotting = True
slicing = False
zak = False
patch_euler_class = False
saving = True
finite_geometry = False
plot_from_save = True


# PARAMETERS
delta_A = -2.5
delta_C = -2.5
omega = 11
A_x = 1
A_y = 1
dJ1 = 0
dJ2 = -0.4

delta_B = -delta_A - delta_C

a_1 = np.array([1,0])
a_2 = np.array([0,1])
num_points = 100
num_steps = 100
num_lines = 100
lowest_quasi_energy = -np.pi / 4
offsets = np.zeros((3,2))


### Zak phase parameters
start = np.array([0.5,0])
end = np.array([0.5,1])

### Euler class parameters
kxmin= 0.4*np.pi
kxmax= 1.6*np.pi
kymin = 0.8*np.pi
kymax = 1.2*np.pi
bands = [1,2]

### Finite geometry parameters
L = 30
cut = 'x'


### Plotting parameters
bands_to_plot = [1,1,1]
node_threshold = 0.05














################################################################################
################################################################################

# CREATING A DIRECTORY
name = 'square_driven_Ax_Ay_w_dJ1_dJ2_dA_dC_{A_x}_{A_y}_{omega}_{dJ1}_{dJ2}_{delta_A}_{delta_C}'.format(A_x=round(A_x,1),A_y=round(A_y,1),
                                                                                                        omega=round(omega,1),dJ1=round(dJ1,1),
                                                                                                        dJ2=round(dJ2,1),delta_A=round(delta_A,1),
                                                                                                        delta_C=round(delta_C,1)).replace('.','p')
main_directory = 'figures/square/all'
directory = main_directory + '/' + name
if saving:
    if not os.path.isdir(directory):
        os.mkdir(directory)



# BLOCH HAMILTONIAN
H = square_hamiltonian_driven(delta_A, delta_B, delta_C, J_ab_0=1, J_ac_0=1, 
                              J_bc_0=1, J_ac_1x=1+dJ1, J_bc_1y=1+dJ1, 
                              J_ab_2m=1+dJ2, A_x=A_x, A_y=A_y, omega=omega)



# BANDSTRUCTURE
if plotting:
    energies, blochvectors = compute_bandstructure2D(H, a_1, a_2, num_points, 
                                                     omega, num_steps, 
                                                     lowest_quasi_energy)
    save = directory + '/' + name + '_structure.png'
    plot_bandstructure2D(energies, a_1, a_2, save, 
                         bands_to_plot=bands_to_plot, 
                         lowest_quasi_energy=lowest_quasi_energy, 
                         discontinuity_threshold = 0.05)

    if saving:
        grid_directory = directory + '/grids'
        if not os.path.isdir(grid_directory):
            os.mkdir(grid_directory)
        energy_save = grid_directory + '/' + name +'_energies'
        vector_save = grid_directory + '/' + name +'_vectors'
        np.save(energy_save, energies)
        np.save(vector_save, blochvectors)



# ZAK PHASE
if zak:
    zak_phase, energies = compute_zak_phase(H, a_1, a_2, offsets, start, end, 
                                            num_points, omega, num_steps, 
                                            lowest_quasi_energy)
    zak_phase = np.rint(np.real(zak_phase / np.pi))
    print(zak_phase)
    fig = plt.figure()
    plt.plot(energies[:,0], label='band 0')
    plt.plot(energies[:,1], label='band 1')
    plt.plot(energies[:,2], label='band 2')
    plt.legend()
    plt.title('{zak_phase}'.format(zak_phase=zak_phase))
    plt.xlabel('$k$')
    plt.ylabel('$E / J$')
    plt.show()
    plt.close()



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



# PATCH EULER CLASS
if patch_euler_class:
    chi = compute_patch_euler_class(kxmin,kxmax,kymin,kymax,bands,H,num_points,omega,num_steps,lowest_quasi_energy)
    print(chi)


# FINITE GEOMETRY
if finite_geometry:
    if cut == 'x':
        H = square_hamiltonian_driven_finite_x(
            L=L,
            delta_a=delta_A,
            delta_b=delta_B,
            delta_c=delta_C,
            J_ab_0=1,
            J_ac_0=1,
            J_bc_0=1,
            J_ac_1x=1+dJ1,
            J_bc_1y=1+dJ1,
            J_ab_2m=1+dJ2,
            A_x=A_x,
            A_y=A_y,
            omega=omega
            )
        
    elif cut == 'y':
        H = square_hamiltonian_driven_finite_y(
            L=L,
            delta_a=delta_A,
            delta_b=delta_B,
            delta_c=delta_C,
            J_ab_0=1,
            J_ac_0=1,
            J_bc_0=1,
            J_ac_1x=1+dJ1,
            J_bc_1y=1+dJ1,
            J_ab_2m=1+dJ2,
            A_x=A_x,
            A_y=A_y,
            omega=omega
            )

    k = np.linspace(-np.pi, np.pi, num_points)
    E = np.zeros((num_points, 3*L))
    blochvectors = np.zeros((num_points, 3*L, 3*L), dtype='complex')

    for i in range(len(k)):
        energies, eigenvectors = compute_eigenstates(H,k[i],omega,num_steps,
                                                    lowest_quasi_energy,False,method='Runge-Kutta')
        E[i] = energies
        blochvectors[i] = eigenvectors

    E, blochvectors = sort_energy_path(E,blochvectors)

    plt.plot(k,E,c='0')
    plt.ylabel('$2\pi E / \omega$')
    plt.yticks([-2*np.pi, -3/2*np.pi, -np.pi, -1/2*np.pi, 0, 1/2*np.pi, np.pi, 
                3/2*np.pi, 2*np.pi], ['$-2\pi$','$-3/2\pi$','$-\pi$','$-1/2\pi$',
                                    '$0$','$1/2\pi$','$\pi$','$3/2\pi$','$2\pi$'])
    if cut == 'x':
        plt.xlabel('$k_y$')
    elif cut == 'y':
        plt.xlabel('$k_x$')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','','0','','$\pi$'])
    plt.xlim(-np.pi, np.pi)
    plt.ylim(lowest_quasi_energy, lowest_quasi_energy + 2*np.pi)
    plt.show()
    if saving:
        finite_dir = directory + '/' + 'finite_geometry'
        if not os.path.isdir(finite_dir):
            os.mkdir(finite_dir)
        energy_save = finite_dir + '/' + name + '_finite_geometry_energies_' + cut 
        vector_save = finite_dir + '/' + name + '_finite_geometry_vectors_' + cut
        np.save(energy_save, E)
        np.save(vector_save, blochvectors)
        plot_save = finite_dir + '/' + name + 'finite_geometry_bands_' + cut + '.png'
        plt.savefig(plot_save)
    plt.close()



if plot_from_save:
    file = directory + '/grids/' + name +'_energies.npy'
    energies = np.load(file)
    #Getting the energies in the right range
    energies = (energies + 2*np.pi*np.floor((lowest_quasi_energy-energies) 
                                                / (2*np.pi) + 1))
    plot_bandstructure2D(energies, a_1, a_2, 'test.png', bands_to_plot=bands_to_plot, lowest_quasi_energy=lowest_quasi_energy)
    node_save = directory + '/' + name + '_nodes.png'
    locate_nodes(energies, a_1, a_2, node_save, node_threshold=node_threshold, title=name)


