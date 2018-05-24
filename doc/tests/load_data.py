import pandas as pd
import numpy as np

def load_data(filename):
    """The function to do the primary data clean process on the input data file."""
    data = pd.read_csv(filename)
    z = data.iloc[:, 0]

    z_approach = z[: len(z)//2]
    z_retractt = z[len(z)//2:]

    z_approach = z_approach.reset_index(drop=True)
    z_retract = z_retractt.reset_index(drop=True)
    return data, z, z_approach, z_retract