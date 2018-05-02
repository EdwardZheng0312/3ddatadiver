import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import create_pslist


def threeD_plot(filename, Z_direction, x_actual, y_actual, x_size, y_size):
    """3D plot for the phase shift of the AFM data."""
    if Z_direction = "up":
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[2]
    else:
        Z_dir = create_pslist.create_pslist(filename, x_size, y_size)[1]

    a = np.linspace(0, x_actual, x_size)
    b = np.linspace(0, y_actual, y_size)
    c = Z_dir
    x, z, y = np.meshgrid(a, c, b)

    #phaseshift information
    psasas = []
    for k in range(len(c)):
        for i in range(len(a)):
            for j in range(len(b)):
                A = create_pslist.create_pslist(filename, x_size, y_size)[0][k][i][j]
                psasas.append(A)
    l = psasas

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=l, alpha=0.4)
    ax.set_xlabel('X(nm)', fontsize=15)
    ax.set_ylabel('Y(nm)', fontsize=15)
    ax.set_zlabel('Z(nm)', fontsize=15)
    ax.set_title('3D Plot for the AFM Phase Shift of XXX', fontsize=20)
    plt.show()
    return
