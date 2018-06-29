import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import create_pslist
import load_data


def y_slicing(filename, Z_direction, Y, x_actual, y_actual, x_size, y_size):
    """Y_Slicing function with different input Y."""
    if Z_direction == "up":
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[2]
    else:
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[1]
        
    j = Y
    a = np.linspace(0, x_actual, x_size)
    b = np.linspace(0, y_actual, y_size)[j]
    c = Z_dir
    x, z, y = np.meshgrid(a, c, b)

    psasas = []
    for k in range(len(c)):
        for i in range(len(a)):
            A = (pd.DataFrame(create_pslist.create_pslist(filename, x_size, y_size)[0][k]).transpose().iloc[j])[i]
            psasas.append(A)
    l = psasas

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=l, alpha=0.4)
    ax.set_ylim(top=y_size, bottom=0)
    ax.set_xlabel('X(nm)', fontsize=15)
    ax.set_ylabel('Y(nm)', fontsize=15)
    ax.set_zlabel('Z(nm)', fontsize=15)
    ax.set_title('Y Axis Slicing for the AFM Phase Shift of XXX', fontsize=20)
    plt.show()
    return

