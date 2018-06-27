import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import loaddata
import createpslist
import zdirection
from mpl_toolkits.mplot3d import Axes3D


@image_comparison(baseline_images=['22slicing_Yslices.png'])
def testthreedyslicings():
    z_direction, Z_dir = zdirection.zdirection("PHASEdata.csv", "down")
    a = np.linspace(0, x_actual, x_size)
    b = np.linspace(0, y_actual, y_size)[22]
    c = Z_dir
    X, Z, Y = np.meshgrid(a, c, b)

    Bs = np.array(createpslist.createpslist("PHASEdata.csv",48, 48))[0:len(Z_dir), :, 22].flatten()

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
