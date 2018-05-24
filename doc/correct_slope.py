def correct_slope(3D_array):
    """Function that corrects for sample tilt

    Input: 3D numpy array of a single signal feed.  For instance, the entire dataset of
    zsensor data.  The function is flexible enough to work on any raw data signal, but can
    only work on one at a time.

    Output: 3D numpy array with values adjusted for sample tilt."""

    # We have create an numpy array of the correct shape to populate.
    array_max = np.zeros((len(3D_array[1,:,1]),len(3D_array[1,1,:]))
    #Populate zero arrays with max z values at all x,y positions.
    for j in range(len(3D_arrray[1,:,1])):
        for i in range(len(3D_arrray[1,1,:])):
            array_max[j,i] = max(3D_array[:,i,j])
    #Find the difference between the max and mean values in the z-direction for
    #each x,y point. Populate new matrix with corrected values.

    driftx = np.zeros(len(3D_array[1,:,1]))
    drifty = np.zeros(len(3D_array[1,1,:]))
    corrected_array = np.zeros((len(3D_array[1,:,1]),len(3D_array[1,1,:]))
    #Correct the for sample tilt along to the y-direction, then correct for sample tilt
    #along the x-direction.
    for j in range(len(3D_arrray[1,:,1])):
        for i in range(len(3D_arrray[1,1,:])):
            drifty[j] = np.mean(array_max[j, :])
            driftx[i] = np.mean(corrected_array[:, i])
            corrected_array[j, :] = array_max[j, :] - drifty[j]
            corrected_array[:, i] = corrected_array[:, i] - driftx[i]
    return corrected_array