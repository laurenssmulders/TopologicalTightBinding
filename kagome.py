import numpy as np
from tight_binding.topology import compute_patch_euler_class
from tight_binding.hamiltonians import kagome_hamiltonian_static

H = kagome_hamiltonian_static(0,0,0,1,1)
chi = compute_patch_euler_class(-0.1,0.1,-0.1,0.1,[0,1],H,100,0,0,0,0,'static')
print(chi)