import h5py as h5
import numpy as np

def generate_array(target):
    """Function to pull single dataset from FFM object and initial formatting.  load_h5 function
    must be run prior to generate_array.

    :param target: Name of single dataset given in list keys generated from load_h5 function.
    target parameter must be entered as a string.

    Output: formatted numpy array of a single dataset.

    Example:

    Phase = generate_array('Phase')

    print(Phase[3,3,3]) = 106.05377
    """
    target = np.array(FFM[target])
    if len(target[:,1,1]) < len(target[1,1,:]):
      target = target.transpose(2,0,1)
    return target