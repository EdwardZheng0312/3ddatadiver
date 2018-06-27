import pandas as pd
import numpy as np
import loaddata


def zdirection(filename,z_direction):
    """
    The function is to choose the direction up or down. So we will call different data to get different plots.
    """
    data, z, z_approach, z_retract = loaddata.loaddata(filename)
    
    #Test whether the four data type are correct.#
    assert type(data) == pd.DataFrame, "data type error, pd.DataFrame expected"
    assert type(z) == pd.core.series.Series, "z type error, pd.core.series.Series expected"
    assert type(z_approach) == pd.core.series.Series, "z_approach type error, pd.core.series.Series expected"
    assert type(z_retract) == pd.core.series.Series, "z_retract type error, pd.core.series.Series expected"
    
    if z_direction == "up":
        Z_dir = z_retract
    else:
        Z_dir = z_approach
    assert type(z_direction) == str, "z_direction type error, string expected"
    
    return z_direction, Z_dir

