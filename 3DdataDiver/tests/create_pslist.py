import numpy as np
import pandas as pd

import load_data


def create_pslist(filename, x_size, y_size):
    """The function for reshape the input data file depends on certain shape of the input data file"""
    data, z, z_approach, z_retract = load_data.load_data(filename)
    pslist = []
    for k in range(len(z)):
        phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
        ps = np.array(phaseshift)
        ps_reshape = np.reshape(ps, (x_size, y_size))
        pslist.append(ps_reshape)
    return pslist

