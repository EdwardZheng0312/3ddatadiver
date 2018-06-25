import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import load_data
import create_pslist
import Z_direction
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.testing.decorators import image_comparison
%matplotlib inline

def threeDZ_slicings(filename, z_dir, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
    z_direction, Z_dir = Z_direction.Z_direction(filename, z_dir)
    phaseshift = (create_pslist.create_pslist(filename,x_size, y_size))[location_slices-1].flatten()

    a = np.linspace(0, x_actual, x_size)
    b = np.linspace(0, y_actual, y_size)
    X, Z, Y = np.meshgrid(a, Z_dir[(location_slices)-1], b)
    l = phaseshift

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(X, Y, Z, c=l, s=6, vmax=np.array(create_pslist.create_pslist(filename, x_size, y_size)).max(), vmin=np.array(create_pslist.create_pslist(filename, x_size, y_size)).min())
    cbar = plt.colorbar(im)
    #cbar.set_label(str(valu))
    ax.set_xlim(left=0, right=x_actual)
    ax.set_ylim(bottom=0, top=y_actual)
    ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
    ax.set_xlabel('X(nm)', fontsize=12)
    ax.set_ylabel('Y(nm)', fontsize=12)
    ax.set_zlabel('Z(nm)', fontsize=12)
    ax.set_title('3D Z Slicing (Z='+str(round(Z_dir[(location_slices)-1],4)) + 'nm) for the phaseshift of AFM data', fontsize=13)
