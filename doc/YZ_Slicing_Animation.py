import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation

import load_data
import create_pslist


def yz_slicing_animation(filename, Z_direction, number_slices, x_actual, y_actual, x_size, y_size):
    """YZ Slicing Animation function depend on the number of slices input"""
    if Z_direction == "up":
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[2]
    else:
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(left=0, right=x_size)
    ax.set_ylim(top=y_size, bottom=0)
    ax.set_zlim(bottom=Z_dir.min(), top=Z_dir.max())
    ax.set_xlabel('X(nm)', fontsize=15)
    ax.set_ylabel('Y(nm)', fontsize=15)
    ax.set_zlabel('Z(nm)', fontsize=15)
    ax.set_title('YZ Slicing Animation for the AFM Phase Shift', fontsize=20)
    
#----------------------------------------------------------------------------------------------------------------
    ims = []
    for add in np.arange(number_slices):
        a = np.linspace(0, x_actual, x_size)[add*(x_size//number_slices)]
        b = np.linspace(0, y_actual, y_size)
        c = Z_dir
        x, z, y = np.meshgrid(a,c,b)
        psasas = []
        for k in range(len(c)):
            for j in range(len(b)):
                B = (pd.DataFrame(create_pslist.create_pslist(filename, x_size, y_size)[0][k]).iloc[add*(x_size//number_slices)])[j]
                psasas.append(B)
        l = psasas
        ims.append((ax.scatter(x, y, z, c=l, s=6)))

    im_ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    plt.show()
    im_ani.save('YZ Slice.htm', metadata={'artist':'Guido'})
    return

