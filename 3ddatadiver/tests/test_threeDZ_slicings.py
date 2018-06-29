import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import load_data
import create_pslist
import Z_direction
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.testing.decorators import image_comparison
%matplotlib inline

@image_comparison(baseline_images=['202_down_Zslices.png'])
def test_threeDZ_slicings():
    z_direction, Z_dir = Z_direction.Z_direction("PHASEdata.csv", "down")
    phaseshift = (create_pslist.create_pslist("PHASEdata.csv",48, 48))[202].flatten()

    a = np.linspace(0, 2, 48)
    b = np.linspace(0, 2, 48)
    X, Z, Y = np.meshgrid(a, Z_dir[(202)], b)
    l = phaseshift

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(X, Y, Z, c=l, s=6, vmax=np.array(create_pslist.create_pslist("PHASEdata.csv",48, 48)).max(), vmin=np.array(create_pslist.create_pslist("PHASEdata.csv",48, 48)).min())
    cbar = plt.colorbar(im)
    #cbar.set_label(str(valu))
    ax.set_xlim(left=0, right=2)
    ax.set_ylim(bottom=0, top=2)
    ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
    ax.set_xlabel('X(nm)', fontsize=12)
    ax.set_ylabel('Y(nm)', fontsize=12)
    ax.set_zlabel('Z(nm)', fontsize=12)
    ax.set_title('3D Z Slicing (Z='+str(round(Z_dir[202],4)) + 'nm) for the phaseshift of AFM data', fontsize=13)
    
