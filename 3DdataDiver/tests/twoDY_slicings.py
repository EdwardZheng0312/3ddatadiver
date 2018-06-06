
import pandas as pd
import numpy as np
import load_data
import create_pslist
import Z_direction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


def twoDY_slicings(filename, z_dir, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
    z_direction, Z_dir = Z_direction.Z_direction(filename, z_dir)
    a = np.linspace(0, x_actual, x_size)
    b = np.linspace(0, y_actual, y_size)[location_slices]
    c = Z_dir
    X, Z, Y = np.meshgrid(a, c, b)

    As = np.array(create_pslist.create_pslist(filename, x_size, y_size))[0:len(Z_dir), :, location_slices]

    fig1 = plt.figure(figsize=(11, 9))
    plt.subplot(111)
    plt.imshow(As, aspect='auto')
    plt.xlabel('Y', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.title('2D Y Slicing (X='+str(round(b,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)
    plt.colorbar()

    setStr = '{}_2d_Yslices.png'.format(export_filename2)
    fig1.savefig(setStr)