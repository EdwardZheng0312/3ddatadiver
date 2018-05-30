# 3DdataDiver

<img src="https://raw.githubusercontent.com/EdwardZheng0312/VisualanalysisAFM/master/doc/deep-diver.png" width="165" height="229">

3DdataDiver is a package for the processing and visualization of 3D AFM data.  Users interact with the package primarily through
a GUI that is locally run.  Upon uploading .mat or .HDF5 files hidden data cleaning functions are called that correct sample slope,
linearize Zsensor data, concat Zsensor with phase/amp data, and generate a three numpy arrays: full, approach, and retract dataset.
Users can interact with the data by viewing a full 3D rending, slices in 3D or 2D cartesian coordinate systems, and animations of
these slices.  At any point the user can save a .csv file of the data they are currently viewing to the folder the GUI is being
run.

# Use Cases
1. Pipeline for easy datacleaning
	* Process raw data into .csv files for use in the GUI visualizations, or save files to share with your friends!  Supported
	by data_cleaning component.
2. Visualize 3D AFM data
	* View full data set or slices in XY, XZ, or YZ planes.  Sliced data can be quickly viewed in 2D or 3D rendings.  Animation
	option availible in the 3D renderings. Supported by threeD_plot, twoD_slicing, and  animation components.   
