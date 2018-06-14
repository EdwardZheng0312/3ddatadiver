def correct_slope(threeD_array):
    """Function that corrects for sample tilt and generates arrays used in the bin_array
    function.

    :param threeD_array: 3D numpy array of a single signal feed.  For instance, the entire dataset of
    zsensor data.  The function is flexible enough to work on any raw data signal, but can
    only work on one at a time.

    Output: 3D numpy array with values adjusted for sample tilt, a 2D numpy array with
    the index of the max Zsensor value for each x,y coordinate, 1D numpy array of the X
    direction correction, and 1D numpy array of the Y direction correction.

    Example of function calling format:
    ZSNSRtotCORR, indZ, driftx, drifty = correct_slope(zsensor)"""

    #Convert matrix from meters to nanometers.
    threeD_array = np.multiply(threeD_array, -1000000000)
    #Replace zeros and NaN in raw data with neighboring values.  Interpolate does not work
    #as many values are on the edge of the array.
    threeD_array[threeD_array == 0] = np.nan
    mask = np.isnan(threeD_array)
    threeD_array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), threeD_array[~mask])
    # We have create an numpy array of the correct shape to populate.
    array_min = np.zeros((len(threeD_array[1,:,1]),len(threeD_array[1,1,:])))
    indZ = np.zeros((len(threeD_array[1,:,1]),len(threeD_array[1,1,:])))
    #Populate zero arrays with min z values at all x,y positions.  Also, populate indZ array
    #with the index of the min z values for use in correct_Zsnsr()
    for j in range(len(threeD_array[1,:,1])):
        for i in range(len(threeD_array[1,1,:])):
            array_min[j,i] = np.min(threeD_array[:,i,j])
            indZ[j, i] = np.min(np.where(threeD_array[:, i, j] == np.min(threeD_array[:, i, j])))
    #Find the difference between the max and mean values in the z-direction for
    #each x,y point. Populate new matrix with corrected values.

    driftx = np.zeros(len(threeD_array[1,:,1]))
    drifty = np.zeros(len(threeD_array[1,1,:]))
    corrected_array = np.zeros((len(threeD_array[1,:,1]),len(threeD_array[1,1,:])))
    #Correct the for sample tilt along to the y-direction, then correct for sample tilt
    #along the x-direction.
    for j in range(len(threeD_array[1,:,1])):
        for i in range(len(threeD_array[1,1,:])):
            drifty[j] = np.mean(array_min[j, :])
            driftx[i] = np.mean(corrected_array[:, i])
            corrected_array[j, :] = array_min[j, :] - drifty[j]
            corrected_array[:, i] = corrected_array[:, i] - driftx[i]

    #Apply corrected slope to each level of 3D numpy array
    arraytotcorr = np.empty_like(threeD_array)
    for j in range(len(threeD_array[1, :, 1])):
        for i in range(len(threeD_array[1, 1, :])):
            arraytotcorr[:, i, j] = threeD_array[:, i, j] - driftx[i] - drifty[j]
    return arraytotcorr, indZ
