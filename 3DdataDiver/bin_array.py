def bin_array(arraytotcorr, indZ, rawarray):
    """
    Function to reduce the size of large datasets.  Data placed into equidistant bins for each x,y coordinate and new
    vector created from the mean of each bin.  Size of equidistant bins determined by 0.01 nm increments of Zsensor
    data.
    :param arraytotcorr: Zsensor data corrected for sample tilt using correct_slope function.  Important to use this
     and not raw Zsensor data so as to get an accurate Zmax value.
    :param indZ: Index of Zmax for each x,y coordinate to cut data set into approach and retract.
    :param rawarray: 3D numpy array the user wishes to reduce in size (e.g. phase, amp)
    :return: 3D numpy array of binned approach values, 3D numpy array of binned retract values.

    """
    #Generate empty numpy arrays to populate.
    arraymean = np.zeros(len(arraytotcorr[:,1,1]))
    reduced_array_approach = np.zeros((len(linearized),len(arraytotcorr[1,:,1]),len(arraytotcorr[1,1,:])))
    reduced_array_retract = np.zeros((len(linearized),len(arraytotcorr[1,:,1]),len(arraytotcorr[1,1,:])))
    #Create list of the mean Zsensor value for each horizontal slice of Zsensor array.
    for z in range(len(arraymean)):
        arraymean[z] = np.mean(arraytotcorr[z, :, :])
    #Turn mean Zsensor data into a linear vector with a step size of 0.01 nm.
    linearized = np.flip(np.arange(0, np.max(arraymean), (0.01)), axis=0)
    #Replace zeros and NaN in raw data with neighboring values.  Interpolate does not work
    #as many values are on the edge of the array.
    rawarray[rawarray == 0] = np.nan
    mask = np.isnan(rawarray)
    rawarray[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), rawarray[~mask])
    #Cut raw phase/amp datasets into approach and retract, then bin data according to the linearized Zsensor data.
    #Generate new arrays from the means of each bin.
    for j in range(len(arraytotcorr[1, :, 1])):
        for i in range(len(arraytotcorr[1, 1, :])):
            z = rawarray[:(int(indZ[i, j])), i, j]
            bin_means = (np.histogram(z, len(linearized), weights=z)[0] / np.histogram(z, len(linearized))[0])
            reduced_array_approach[:, i, j] = bin_means.flatten()
    for j in range(len(arraytotcorr[1, :, 1])):
        for i in range(len(arraytotcorr[1, 1, :])):
            z = rawarray[-(int(indZ[i, j])):, i, j]
            bin_means = (np.histogram(z, len(linearized), weights=z)[0] / np.histogram(z, len(linearized))[0])
            reduced_array_retract[:, i, j] = bin_means.flatten()
    return reduced_array_approach, reduced_array_retract