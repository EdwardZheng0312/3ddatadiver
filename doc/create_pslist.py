
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import load_data

def create_pslist(filename,x_size,y_size):
    data,z,z_approach,z_retract = load_data.load_data(filename)
    #phase shift
    pslist = []
    for k in range(len(z)):
        phaseshift = data.iloc[k,1:]  #[from zero row to the end row, from second column to the last column]
        ps = np.array(phaseshift)
        ps_reshape = np.reshape(ps,(x_size,y_size))
        pslist.append(ps_reshape)
    return pslist,z_approach,z_retract

