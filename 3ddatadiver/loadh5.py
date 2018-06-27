import h5py as h5


def showkeys(dataset):
    """Function to load .h5 files and print the corresponding dataset names.

    :param dataset:  File name and relative path if file is located in a different directory
    than 3DdataDiver is being run from.  dataset parameter must be entered as a string.

    Output:  List of .h5 files keys and the full file loaded to an object and the raw datasets
     loaded to an object.  Please use the following format to ensure the code in the rest of
     package runs correctly:

    f,FFM = show_keys('dataset')

    f and FFM objects are called explicitly in later code."""
    f = h5.File(dataset, "r+")
    FFM = f['FFM']
    print(list(FFM.keys()))
    return f,FFM





