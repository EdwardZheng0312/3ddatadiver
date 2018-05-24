
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import load_data

def Z_direction(filename,z_direction):
    data, z, z_approach, z_retract = load_data.load_data(filename)
    if z_direction == "up":
        Z_dir = z_retract
    else:
        Z_dir = z_approach
    return z_direction, Z_dir

