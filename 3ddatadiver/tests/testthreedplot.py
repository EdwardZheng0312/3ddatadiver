import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import loaddata
import createpslist


@image_comparison(baseline_images=['3D Plot.png'])
def testthreedplot():
    data, z, z_approach, z_retract = loaddata.loaddata(filename)
    Z_dir = data.iloc[:,0].iloc[-len(z) // 2:]
    data1 = data.iloc[:,:].iloc[-len(z) // 2:].drop(['Z (nm)'], axis=1)
    
    retract_as_numpy = data1.as_matrix(columns=None)
    retract_as_numpy_reshape1 = retract_as_numpy.reshape(len(Z_dir), x_size, y_size)
    retract_as_numpy_reshape2 = retract_as_numpy_reshape1.flatten('F')

        # Code for the plotting
    x = np.linspace(0, x_actual, x_size)
    y = np.linspace(0, y_actual, y_size)
    z = np.linspace(0, Z_dir.max(), len(Z_dir))
    x,y,z = np.meshgrid(x,y,z)
    
    fxyz = list(retract_as_numpy_reshape2)
      

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(x, y, z, c=fxyz, alpha=0.2)
    plt.colorbar(im)
    ax.set_xlim(left=0,right=x_actual)
    ax.set_ylim(top=y_actual, bottom=0)
    ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
    ax.set_xlabel('X(nm)', fontsize=15)
    ax.set_ylabel('Y(nm)', fontsize=15)
    ax.set_zlabel('Z(nm)', fontsize=15)
    ax.set_title('3D Plot for Phase Shift of the AFM data', fontsize=20)

