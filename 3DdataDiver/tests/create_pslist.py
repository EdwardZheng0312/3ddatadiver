import numpy as np
import pandas as pd

import load_data


def create_pslist(filename, x_size, y_size):
    """The function for reshape the input data file depends on certain shape of the input data file"""
    data, z, z_approach, z_retract = load_data.load_data(filename)
    #Test whether the four data type are correct.#
    assert type(data) == pd.DataFrame, "data type error, pd.DataFrame expected"
    assert type(z) == pd.core.series.Series, "z type error, pd.core.series.Series expected"
    assert type(z_approach) == pd.core.series.Series, "z_approach type error, pd.core.series.Series expected"
    assert type(z_retract) == pd.core.series.Series, "z_retract type error, pd.core.series.Series expected"
    
    pslist = []
    for k in range(len(z)):
        phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
        ps = np.array(phaseshift)
        assert type(ps) == np.array, "ps type error, array expected"
        
        ps_reshape = np.reshape(ps, (x_size, y_size))
        pslist.append(ps_reshape)
    #Test the pslist type #
    assert type(pslist) == list, "pslist type error, list expected"
    
    return pslist

