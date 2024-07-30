import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

r = 1
c = 1
num_points = 100

# DEFINING TWO BANDS
print('Defining bands...')
k = np.linspace(0,1,num_points)
kx, ky = np.meshgrid(k,k,indexing='ij')
E1 = np.zeros((num_points,num_points),dtype='float')
E2 = np.zeros((num_points,num_points),dtype='float')

## setting a constant background
E1 += 1
E2 += -1

radius = 0.2
centres = np.array([[0.25,0.25],[0.75,0.75]])
diff = 2

def energy_difference(k,radius,centres,diff):
    # finding the closest node
    distances = np.zeros((centres.shape[0]))
    for i in range(centres.shape[0]):
        distance = centres[i] - k
        for j in range(distance.shape[0]):
            options = np.array([distance[j],1-abs(distance[j])])
            distance[j] = np.min(np.abs(options))
        distances[i] = np.linalg.norm(distance)
    return diff / 2 * np.exp(-distances[np.argmin(distances)]/radius)

for i in range(num_points):
    for j in range(num_points):
        k = np.array([kx[i,j],ky[i,j]])
        E1[i,j] -= energy_difference(k,radius,centres,diff)
        E2[i,j] += energy_difference(k,radius,centres,diff)


# PLOTTING
print('Plotting...')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(kx, ky, E1, cmap=cm.YlOrRd, edgecolor='darkred',
                        linewidth=0, rstride=r, cstride=c)
surf2 = ax.plot_surface(kx, ky, E2, cmap=cm.PuRd, edgecolor='purple',
                        linewidth=0, rstride=r, cstride=c)

# Get rid of colored axes planes
# First remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
ax.zaxis._axinfo["grid"].update({"linewidth":0.5})

plt.show()
plt.close()
    