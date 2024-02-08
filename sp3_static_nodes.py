import matplotlib.pyplot as plt
import numpy as np

from tight_binding.bandstructure import compute_bandstructure2D
from tight_binding.utilitities import compute_reciprocal_lattice_vectors_2D
from tight_binding.hamiltonians import square_hamiltonian_static

from matplotlib.widgets import Button, Slider

a_1 = np.array([1,0])
a_2 = np.array([0,1])
num_points = 50

def find_nodes(energy_grid,
                node_threshold = 1,
                kxmin=-np.pi, 
                kxmax=np.pi, 
                kymin=-np.pi, 
                kymax=np.pi):
    # Need to periodically extend the energy array to span the whole region
    b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)
    num_points = energy_grid.shape[0]
    dim = energy_grid.shape[2]
    span = False
    copies = 0
    while not span:
        copies += 1
        alpha = np.linspace(-copies,copies,2*copies*num_points,endpoint=False)
        alpha_1, alpha_2 = np.meshgrid(alpha, alpha, indexing = 'ij')
        kx = alpha_1 * b_1[0] + alpha_2 * b_2[0]
        ky = alpha_1 * b_1[1] + alpha_2 * b_2[1]
        span = ((np.min(kx) < kxmin) and (np.max(kx) > kxmax) 
                    and (np.min(ky) < kymin) and (np.max(ky) > kymax))
        
    # Specifying which indices in the original array correspond to indices in 
    # the extended array
    i = ((alpha_1%1) * num_points).astype(int)
    j = ((alpha_2%1) * num_points).astype(int)
    energy_grid_extended = energy_grid[i,j]
    E = np.transpose(energy_grid_extended, (2,0,1))

    gap_1 = abs((E[1] - E[0]))
    gap_2 = abs((E[2] - E[1]))
    gap_3 = abs((E[0] - E[2]))


    gap_1 = gap_1 < node_threshold
    gap_2 = gap_2 < node_threshold
    gap_3 = gap_3 < node_threshold

    gap_1_nodes_kx = []
    gap_2_nodes_kx = []
    gap_3_nodes_kx = []
    gap_1_nodes_ky = []
    gap_2_nodes_ky = []
    gap_3_nodes_ky = []
    

    for i in range(gap_1.shape[0]):
        for j in range(gap_1.shape[1]):
            if gap_1[i,j]:
                gap_1_nodes_kx.append(kx[i,j])
                gap_1_nodes_ky.append(ky[i,j])
    
    for i in range(gap_2.shape[0]):
        for j in range(gap_2.shape[1]):
            if gap_2[i,j]:
                gap_2_nodes_kx.append(kx[i,j])
                gap_2_nodes_ky.append(ky[i,j])
    
    for i in range(gap_3.shape[0]):
        for j in range(gap_3.shape[1]):
            if gap_3[i,j]:
                gap_3_nodes_kx.append(kx[i,j])
                gap_3_nodes_ky.append(ky[i,j])

    return gap_1_nodes_kx, gap_1_nodes_ky, gap_2_nodes_kx, gap_2_nodes_ky, gap_3_nodes_kx, gap_3_nodes_ky


# Define initial parameters
init_A = 0
init_C = 0

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
H = square_hamiltonian_static(
    delta_a=init_A,
    delta_b=-init_A-init_C,
    delta_c=init_C,
    J_ab_0=1,
    J_ac_0=1,
    J_bc_0=1,
    J_ac_1x=1,
    J_bc_1y=1,
    J_ab_2m=1
)
energies, _ = compute_bandstructure2D(H,a_1,a_2,num_points,regime='static')
gap1_x, gap1_y, gap2_x, gap2_y, gap3_x, gap3_y = find_nodes(energies)
gap1 = ax.scatter(gap1_x, gap1_y, label='Gap 1')
gap2 = ax.scatter(gap2_x, gap2_y, label='Gap 2')
gap3 = ax.scatter(gap3_x, gap3_y, label='Gap 3')
ax.legend()
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(['$-\pi$','','','','$\pi$'])
ax.set_yticklabels(['$-\pi$','','','','$\pi$'])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.5)

# Make a horizontal slider to control the frequency.
axA = fig.add_axes([0.25, 0.1, 0.65, 0.03])
A_slider = Slider(
    ax=axA,
    label='$\Delta_A$',
    valmin=-5,
    valmax=5,
    valinit=init_A,
)

# Make a vertically oriented slider to control the amplitude
axC = fig.add_axes([0.25, 0.05, 0.65, 0.03])
C_slider = Slider(
    ax=axC,
    label="$\Delta_C$",
    valmin=-5,
    valmax=5,
    valinit=init_C
)


# The function to be called anytime a slider's value changes
def update(val):
    ax.clear()
    H = square_hamiltonian_static(
        delta_a=A_slider.val,
        delta_b=-A_slider.val-C_slider.val,
        delta_c=C_slider.val,
        J_ab_0=1,
        J_ac_0=1,
        J_bc_0=1,
        J_ac_1x=1,
        J_bc_1y=1,
        J_ab_2m=1
    )
    energies, _ = compute_bandstructure2D(H,a_1,a_2,num_points,regime='static')
    gap1_x, gap1_y, gap2_x, gap2_y, gap3_x, gap3_y = find_nodes(energies)
    gap1 = ax.scatter(gap1_x, gap1_y, label='Gap 1')
    gap2 = ax.scatter(gap2_x, gap2_y, label='Gap 2')
    gap3 = ax.scatter(gap3_x, gap3_y, label='Gap 3')
    ax.legend()
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(['$-\pi$','','','','$\pi$'])
    ax.set_yticklabels(['$-\pi$','','','','$\pi$'])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)


# register the update function with each slider
A_slider.on_changed(update)
C_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    A_slider.reset()
    C_slider.reset()
button.on_clicked(reset)

plt.show()