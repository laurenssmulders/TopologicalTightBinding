import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from tight_binding.bandstructure import compute_bandstructure2D
from tight_binding.utilitities import compute_reciprocal_lattice_vectors_2D
from tight_binding.hamiltonians import square_hamiltonian_static
from mpl_toolkits.mplot3d import Axes3D


from matplotlib.widgets import Button, Slider

a_1 = np.array([1,0])
a_2 = np.array([0,1])
num_points = 50
kxmin = -np.pi
kxmax = np.pi
kymin = -np.pi
kymax = np.pi

# Define initial parameters
init_A = 0
init_C = 0

# Create the figure and the line that we will manipulate
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
energy_grid, _ = compute_bandstructure2D(H,a_1,a_2,num_points,regime='static')
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

# Masking the data we do not want to plot
E[:, (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan

# Plotting
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.YlGnBu,
                        linewidth=0)
surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.PuRd,
                        linewidth=0)
surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.YlOrRd,
                        linewidth=0)
tick_values = np.linspace(-4,4,9) * np.pi / 2
tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
ax.set_xticks(tick_values)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_values)
ax.set_yticklabels(tick_labels)
ax.set_zlim(np.nanmin(E),np.nanmax(E))
ax.set_xlim(kxmin,kxmax)
ax.set_ylim(kymin,kymax)
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.grid(False)
ax.set_box_aspect([1, 1, 2])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)

# Make a horizontal slider to control the frequency.
axA = fig.add_axes([0.25, 0.1, 0.65, 0.03])
A_slider = Slider(
    ax=axA,
    label='$\Delta_A$',
    valmin=-15,
    valmax=15,
    valinit=init_A,
)

# Make a vertically oriented slider to control the amplitude
axC = fig.add_axes([0.25, 0.05, 0.65, 0.03])
C_slider = Slider(
    ax=axC,
    label="$\Delta_C$",
    valmin=-15,
    valmax=15,
    valinit=init_C
)

# The function to be called anytime a slider's value changes
def update(val):
    ax.clear()
    print('cleared')
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
    energy_grid, _ = compute_bandstructure2D(H,a_1,a_2,num_points,regime='static')
    # Need to periodically extend the energy array to span the whole region
    b_1, b_2 = compute_reciprocal_lattice_vectors_2D(a_1, a_2)
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

    # Masking the data we do not want to plot
    E[:, (kx>kxmax) | (kx<kxmin) | (ky>kymax) | (ky<kymin)] = np.nan

    # Plotting
    surf1 = ax.plot_surface(kx, ky, E[0], cmap=cm.YlGnBu,
                            linewidth=0)
    surf2 = ax.plot_surface(kx, ky, E[1], cmap=cm.PuRd,
                            linewidth=0)
    surf3 = ax.plot_surface(kx, ky, E[2], cmap=cm.YlOrRd,
                            linewidth=0)
    tick_values = np.linspace(-4,4,9) * np.pi / 2
    tick_labels = ['$-2\pi$', '', '$-\pi$', '', '0', '', '$\pi$', '', '$2\pi$']
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values)
    ax.set_yticklabels(tick_labels)
    ax.set_zlim(np.nanmin(E),np.nanmax(E))
    ax.set_xlim(kxmin,kxmax)
    ax.set_ylim(kymin,kymax)
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.grid(False)
    ax.set_box_aspect([1, 1, 2])
    


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