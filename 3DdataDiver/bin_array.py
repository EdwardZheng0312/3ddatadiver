import numpy as np

def bin_array(arraytotcorr, indZ, threeD_array):
    """
    Function to reduce the size of large datasets.  Data placed into equidistant bins for each x,y coordinate and
    new vector created from the mean of each bin.  Size of equidistant bins determined by 0.01 nm increments of
    Zsensor data.
    :param arraytotcorr: Zsensor data corrected for sample tilt using correct_slope function.  Important to use this
     and not raw Zsensor data so as to get an accurate Zmax value.
    :param indZ: Index of Zmax for each x,y coordinate to cut data set into approach and retract.
    :param rawarray: 3D numpy array the user wishes to reduce in size (e.g. phase, amp)
    :return: 3D numpy array of binned approach values, 3D numpy array of binned retract values.
    """
    assert np.isfortran(threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                Needs to be column-major indexing."


    # Generate empty numpy array to populate.
    global reduced_array_approach
    global reduced_array_retract
    global linearized
    arraymean = np.zeros(len(arraytotcorr[:, 1, 1]))
    digitized = np.empty_like(arraymean)
    # Create list of the mean Zsensor value for each horizontal slice of Zsensor array.
    for z in range(len(arraymean)):
        arraymean[z] = np.mean(arraytotcorr[z, :, :])
    # Turn mean Zsensor data into a linear vector with a step size of 0.02 nm.
    linearized = np.arange(-0.2, arraymean.max(), 0.02)
    # Generate empty array to populate
    reduced_array_approach = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
    reduced_array_retract = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
    # Cut raw phase/amp datasets into approach and retract, then bin data according to the linearized Zsensor data.
    # Generate new arrays from the means of each bin.  Perform on both approach and retract data.
    for j in range(len(arraytotcorr[1, :, 1])):
        for i in range(len(arraytotcorr[1, 1, :])):
            z = arraytotcorr[:(int(indZ[i, j])), i, j]  # Create dataset with just retract data
            digitized = np.digitize(z, linearized)  # Bin Z data based on standardized linearized vector.
            for n in range(len(linearized)):
                ind = list(np.where(digitized == n)[0])  # Find which indices belong to which bins
                reduced_array_approach[n, i, j] = np.mean(threeD_array[ind, i, j])  # Find the mean of the bins and
                # populate new array.

    for j in range(len(arraytotcorr[1, :, 1])):
        for i in range(len(arraytotcorr[1, 1, :])):
            z = arraytotcorr[-(int(indZ[i, j])):, i, j]  # Create dataset with just approach data.
            z = np.flipud(z)  # Flip array so surface is at the bottom on the plot.
            digitized = np.digitize(z, linearized)  # Bin Z data based on standardized linearized vector.
            for n in range(len(linearized)):
                ind = list(np.where(digitized == n)[0])  # Find which indices belong to which bins
                reduced_array_retract[n, i, j] = np.mean(threeD_array[ind, i, j])  # Find the mean of the bins and
                # populate new array.

    return linearized, reduced_array_approach, reduced_array_retract