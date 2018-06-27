import pandas as pd
import numpy as np

def loaddata(filename):
    """The function to do the primary data clean process on the input data file."""
    data = pd.read_csv(filename)
    z = data.iloc[:, 0]

    z_approach = z[: len(z)//2]
    z_retractt = z[len(z)//2:]

    z_approach = z_approach.reset_index(drop=True)
    z_retract = z_retractt.reset_index(drop=True)
    
    
    assert type(data) == pd.DataFrame, "data type error, pd.DataFrame expected"
    assert type(z) == pd.core.series.Series, "z type error, pd.core.series.Series expected"
    assert type(z_approach) == pd.core.series.Series, "z_approach type error, pd.core.series.Series expected"
    assert type(z_retract) == pd.core.series.Series, "z_retract type error, pd.core.series.Series expected"
    
    return data, z, z_approach, z_retract