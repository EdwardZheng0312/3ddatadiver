
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import load_data
import create_pslist
import Z_direction

# In[2]:


@image_comparison(baseline_images=['22slicing_2d_Zslices'])
def test_twoZ_slicing():
    z_direction, Z_dir = Z_direction.Z_direction("PHASEdata.csv","up")
    phaseshift = (create_pslist.create_pslist("PHASEdata.csv",48,48))[int(22)]

    l = phaseshift

    fig = plt.figure(figsize=(9, 9))
    plt.imshow(l)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('2D Z Slicing (Z='+str(round(Z_dir[location_slices],3)) + 'nm) for the Phase Shift of AFM data', fontsize=13)
    plt.colorbar()

