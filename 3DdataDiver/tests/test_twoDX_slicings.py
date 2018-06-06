import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import load_data
import create_pslist
import Z_direction
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

@image_comparison(baseline_images=['22_2d_Xslices.png'])
def test_twoDX_slicings():
    z_direction, Z_dir = Z_direction.Z_direction("PHASEdata.csv", "down")
    a = np.linspace(0, x_actual, x_size)[22]
    b = np.linspace(0, y_actual, y_size)
    c = Z_dir
    X, Z, Y = np.meshgrid(a, c, b)

    As = np.array(create_pslist.create_pslist(filename, x_size, y_size))[0:len(Z_dir), 22, :]

    fig1 = plt.figure(figsize=(11, 9))
    plt.subplot(111)
    plt.imshow(As, aspect='auto')
    plt.xlabel('Y', fontsize=12)
    plt.ylabel('Z', fontsize=12)
    plt.title('2D X Slicing (X='+str(round(a,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)
    plt.colorbar()