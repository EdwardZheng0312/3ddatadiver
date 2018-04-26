import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

# ---------------------------------------------------------------------------------------------------------------
data = pd.read_csv('PHASEdata.csv', header=None, skiprows=1)
z = data[0]
# print(z)
z_approach = z[:500]
z_retract = z[500:]
# print(z_approach)
# print(z_retract)

# phase shift
pslist = []
for k in range(len(z)):
    phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
    # print(phaseshift)
    ps = np.array(phaseshift)
    ps_reshape = np.reshape(ps, (48, 48))
    pslist.append(ps_reshape)
# print(pslist)
# ----------------------------------------------------------------------------------------------------------------
for i in range(0, 48):
    a = np.linspace(0, 47, 48)[i]
    b = np.linspace(0, 47, 48)
    c = z_approach
    x, z, y = np.meshgrid(a, c, b)

nFrames = 48

# Set up plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(left=0, right=48)
ax.set_ylim(top=48, bottom=0)
ax.set_zlim(bottom=0, top=2)
ax.set_xlabel('X(nm)', fontsize=15)
ax.set_ylabel('Y(nm)', fontsize=15)
ax.set_zlabel('Z(nm)', fontsize=15)
ax.set_title('X Slicing Plot for the AFM Phase Shift of XXX', fontsize=20)


# Animation function
def animate(i):
    for i in range(0, 48):
        a = np.linspace(0, 47, 48)[i]
        b = np.linspace(0, 47, 48)
        c = z_approach
        x, z, y = np.meshgrid(a, c, b)


    cont = plt.scatter(x, y, z, alpha=0.4)

    return cont


anim = animation.FuncAnimation(fig, animate, frames=nFrames)
plt.show()