def correct_slope(threeD_array):
    """Function that corrects for sample tilt

    Input: 3D numpy array of a single signal feed.  For instance, the entire dataset of
    zsensor data.  The function is flexible enough to work on any raw data signal, but can
    only work on one at a time.

    Output: 3D numpy array with values adjusted for sample tilt."""

    #Convert matrix from meters to nanometers.
    threeD_array = np.multiply(threeD_array, -1000000000)
    # We have create an numpy array of the correct shape to populate.
    array_min = np.zeros((len(threeD_array[1,:,1]),len(threeD_array[1,1,:])))
    indZ = np.zeros((len(threeD_array[1,:,1]),len(threeD_array[1,1,:])))
    #Populate zero arrays with min z values at all x,y positions.  Also, populate indZ array
    #with the index of the min z values for use in correct_Zsnsr()
    for j in range(len(threeD_array[1,:,1])):
        for i in range(len(threeD_array[1,1,:])):
            array_min[j,i] = np.min(threeD_array[:,i,j])
            indZ[j, i] = np.min(np.where(zsensor[:, i, j] == np.min(zsensor[:, i, j])))
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
    return corrected_array, indZ
