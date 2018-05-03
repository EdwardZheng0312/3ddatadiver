
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:
# Call the function like this: pslist,z_approach = data('PHASEdata.csv')

def load_data(filename):
    data = pd.read_csv(filename)
    z = data.iloc[:,0]
    z_approach = z[:500]
    z_retract = z[500:]
    return data,z,z_approach,z_retract
