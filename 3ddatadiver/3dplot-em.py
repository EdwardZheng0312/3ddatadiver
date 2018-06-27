import pandas as pd
import numpy as np

%matplotlib notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

def import_data_retract(file):
    """
    Function to import data file and clean for use in graphing code
    
    import file as CSV!
    
    returns shaped np array (500,48,48)
    
    """
    data = pd.read_csv(file)
    data.rename(columns={'Phase ': 'Unnamed: 1'}, inplace=True) ##this is specific
            ##to the example data might not need in the future.
    i = iter(range(1, len(data.columns) + 1)) ##replace unnamed columns with 
    data.columns = [x if not x.startswith('Unnamed') else next(i) for x in data.columns]
    data = pd.DataFrame(data)
    retract = data[-500:]
    retract = retract.drop(['Z (nm)'], axis=1)
    retract_as_numpy = retract.as_matrix(columns=None)
    retract_as_numpy_reshape = np.array(retract_as_numpy.tolist())
    retract_as_numpy_reshape1 = retract_as_numpy.reshape(500,48,48)
    return retract_as_numpy_reshape1

#This is just me being lazy and not altering my above function, but to run the plotting code below you need to flatten the 3Darray.
retract_as_numpy_reshape2 = retract_as_numpy_reshape1.flatten('F')

#Code for the plotting
x = np.linspace(0,2,48)
y = np.linspace(0,2,48)
z = np.linspace(0,2,500)

#This creates a "flat" list representation of a 3Dspace
points = []
for element in itertools.product(x, y, z):
    points.append(element)

#phase data
fxyz = list(retract_as_numpy_reshape2)
#Not entirely sure what the dealio is here.
xi, yi, zi = zip(*points)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xi, yi, zi, c=fxyz, alpha=0.5)
plt.show()