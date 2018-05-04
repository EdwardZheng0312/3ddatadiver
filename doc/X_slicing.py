import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import create_pslist
import load_data


def x_slicing(filename, Z_direction, X, x_actual, y_actual, x_size, y_size):
    """X_Slicing function with different input X."""
    if Z_direction == "up":
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[2]
    else:
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[1]
        
    i = X
    a = np.linspace(0, x_actual, x_size)[i]
    b = np.linspace(0, y_actual, y_size)
    c = Z_dir
    x, z, y = np.meshgrid(a, c, b)

    psasas = []
    for k in range(len(c)):
        for j in range(len(b)):
            B = (pd.DataFrame(create_pslist.create_pslist(filename, x_size, y_size)[0][k]).iloc[i])[j]
            psasas.append(B)
    l = psasas

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=l, alpha=0.4)
    ax.set_xlim(left=0, right=x_size)
    ax.set_xlabel('X(nm)', fontsize=15)
    ax.set_ylabel('Y(nm)', fontsize=15)
    ax.set_zlabel('Z(nm)', fontsize=15)
    ax.set_title('X Axis Slicing for the AFM Phase Shift of XXX', fontsize=20)
    plt.show()
    return
