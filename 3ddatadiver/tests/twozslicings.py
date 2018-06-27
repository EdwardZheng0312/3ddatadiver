import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import loaddata
import createpslist
import zdirection


def twozslicings(filename,z_direction,location_slices, x_actual, y_actual, x_size, y_size,export_filename2):
    z_direction, Z_dir = zdirection.zdirection(filename,z_direction)
    phaseshift = (createpslist.createpslist(filename, x_size, y_size))[int(location_slices)]

    l = phaseshift

    fig = plt.figure(figsize=(9, 9))
    plt.imshow(l)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('2D Z Slicing (Z='+str(round(Z_dir[location_slices],3)) + 'nm) for the Phase Shift of AFM data', fontsize=13)
    plt.colorbar()

    setStr = '{}_2d_Zslices.png'.format(export_filename2)
    fig.savefig(setStr)


