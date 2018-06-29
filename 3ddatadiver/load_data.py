import pandas as pd

def load_data(filename):
    """The function to do the primary data clean process on the input data file."""
    data = pd.read_csv(filename)
    z = data.iloc[:,0]
    z_approach = z[:500]
    z_retract = z[500:]
    return data, z, z_approach, z_retract
