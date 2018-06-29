import pandas as pd
import numpy as np
import load_data
import create_pslist
import Z_direction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def threeDY_slicings(filename, z_dir, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        z_direction, Z_dir = Z_direction.Z_direction(filename, z_dir)
        a = np.linspace(0, x_actual, x_size)
        b = np.linspace(0, y_actual, y_size)[location_slices]
        c = Z_dir
        X, Z, Y = np.meshgrid(a, c, b)

        Bs = np.array(create_pslist.create_pslist(filename, x_size, y_size))[0:len(Z_dir), :, location_slices].flatten()

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=Bs, s=6)
        plt.colorbar(im)
        ax.set_xlim(left=0, right=x_actual)
        ax.set_ylim(bottom=0, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D Y Slicing (Y='+str(round(b,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)

        setStr = '{}_Yslices.png'.format(export_filename2)
        fig.savefig(setStr)
