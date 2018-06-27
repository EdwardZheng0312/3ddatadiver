import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import loaddata
import createpslist
import zdirection
from mpl_toolkits.mplot3d import Axes3D


def threedxslicings(filename, z_dir, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
    z_direction, Z_dir = zdirection.zdirection(filename, z_dir)
    a = np.linspace(0, x_actual, x_size)[location_slices]
    b = np.linspace(0, y_actual, y_size)
    c = Z_dir
    X, Z, Y = np.meshgrid(a, c, b)

    As = np.array(createpslist.createpslist(filename,x_size, y_size))[0:len(Z_dir), location_slices, :].flatten()

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(X, Y, Z, c=As, s=6)
    plt.colorbar(im)
    ax.set_xlim(left=0, right=x_actual)
    ax.set_ylim(bottom=0, top=y_actual)
    ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
    ax.set_xlabel('X(nm)', fontsize=12)
    ax.set_ylabel('Y(nm)', fontsize=12)
    ax.set_zlabel('Z(nm)', fontsize=12)
    ax.set_title('3D X Slicing (X='+str(round(a,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)

    setStr = '{}_Xslices.png'.format(export_filename2)
    fig.savefig(setStr)


