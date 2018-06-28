import h5py as h5
import numpy as np


def generatearray(valu):
    """Function to pull single dataset from FFM object and perform initial formatting.  The .h5 file must be opened,
    with the FFM group pulled out and the Zsnsr array generated before this function can be ran.
    must be run prior to generate_array.

    :param target: Name of single dataset given in list keys generated from load_h5 function.
    target parameter must be entered as a string.

    Output: formatted numpy array of a single dataset.

    Example:

    Phase = generate_array('Phase')

    print(Phase[3,3,3]) = 106.05377
    """
    #  Code is built for Fortran (column-major) formatted arrays and .h5 files/numpy default to row-major arrays.
    #  We need to transpose the data and then convert it to Fortran indexing (order = "F" command).
    temp = np.array(FFM[valu])
    temp = np.transpose(temp)
    threeD_array = np.reshape(temp, (len(temp[:, 1, 1]), len(temp[1, :, 1]), len(temp[1, 1, :])), order="F")

    Zsnsr_temp = np.array(Zsnsr)
    Zsnsr_temp = np.transpose(Zsnsr_temp)
    Zsnsr_threeD_array = np.reshape(Zsnsr_temp, (len(Zsnsr_temp[:, 1, 1]),
                                                 len(Zsnsr_temp[1, :, 1]), len(Zsnsr_temp[1, 1, :])), order="F")
    return threeD_array, Zsnsr_threeD_array