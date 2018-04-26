import pandas as pd
import numpy as np
#import xlrd
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import animation

data=pd.read_csv('PHASEdata.csv')

z = np.zeros((1000,2304))
for j in range(1000):
    for i in range(2304):
        z[j,i] = data.iloc[j,0]
a=np.linspace(1,48,48)
b=np.linspace(1,48,48)
x,y=np.meshgrid(a,b)

from mpl_toolkits.mplot3d import Axes3D

# Set up plotting
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlim(left=0, right=48)
ax.set_ylim(top=48, bottom=0)
ax.set_zlim(bottom=0, top=2)
ax.set_xlabel('X(nm)', fontsize=10)
ax.set_ylabel('Y(nm)', fontsize=10)
ax.set_zlabel('Z(nm)', fontsize=10)
ax.set_title('XY Slicing Plot', fontsize=15)
ims = []
for add in np.arange(20):
    ims.append((ax.scatter(x, y, zs=z[add*50],c=data.iloc[add*50,1:]),))
im_ani = animation.ArtistAnimation(fig,ims,interval=1000,blit=True)
plt.show()
im_ani.save('XY slice_1.htm', metadata={'artist':'Guido'})

