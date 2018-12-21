# 3ddatadiver

<p align="center">
<img src="https://raw.githubusercontent.com/EdwardZheng0312/VisualanalysisAFM/master/doc/deep-diver.png" width="165" height="229">
</p>
3ddatadiver is a package for the processing and visualization of 3D AFM data.  Users interact with the package primarily through
a GUI that is locally run.  Upon uploading a .H5 files hidden data cleaning functions are called that correct sample slope,
linearize Zsensor or Drive data, concat Zsensor/drive with phase/amp data, and generate a three numpy arrays: full, approach, and retract dataset.  The user can choose to turn off the correct slope function if they would like to visualize raw data.  The process data is saved to an .h5 file to easy use in external software.  Users can interact with the data by viewing a full 3D rending, slices in 3D or 2D cartesian coordinate systems, animations the orthogonal slices, 2D slicing along non-orthogonal directions, and full z-vector data at chosen (x,y) coordinates.  At any point the user can save an .h5 file of the data they are currently viewing to the folder the GUI is being run.

# Use Cases
1. Pipeline for easy datacleaning
	* Process raw data into .csv files for use in the GUI visualizations, or save files to share with your friends!  Supported
	by data_cleaning component.
2. Visualize 3D AFM data
	* View full data set or slices in XY, XZ, or YZ planes.  Sliced data can be quickly viewed in 2D or 3D rendings.  Animation
	option availible using a jupyter notebook. Supported by threeD_plot, twoD_slicing, and  animation components.   
3. Export data of current plot
	* Option to export a .h5 file containing the data of the plot currently being viewed.  Exported dataset is saved in the same
	 directory that the GUI is being run from.	
# Installation
git clone https://github.com/EdwardZheng0312/VisualanalysisAFM  <br />
cd 3ddatadiver <br /> 
python setup.py install <br />
pyton master_Tkinter_GUI.py <br /><br />
**OR** 3ddatadiver/dist/Tkinter_GUI.exe

# GUI design
In  order to help users interact with AFM data  through graphical icons and visual indicators instead of complicated code, we built a graphical user interface (GUI) by Tkinter.
3DdataDiver has four main functions:

(1)Data Cleaning 

(2)Visualization of the approach or retract dataset.  3DdataDiver is flexible enough to handle datasets of different sizes. 

(3)Slicing the dataset in x plane, y plane, z plane, or non-orthogonal planes.  Users can select any layer of interest to see plots in 2D and 3D  cartesian coordinate systems. For the z-slicing the phase shift (or amplitude, etc) of the a selected point is displayed along with the corresponding coordinates.

(4)Exporting of .h5 files generated in any of the aforementioned steps.  This allows users to users to perform their own manipulations on cleaned data and/or sliced data.

## GUI interface
<p align="center">
<img src="https://github.com/EdwardZheng0312/3ddatadiver/blob/master/doc/GUI.PNG">
</p>


# Poster
<p align="center">
<img src="https://github.com/EdwardZheng0312/3ddatadiver/blob/master/doc/3DdataDiver_poster.png">
</p>

<object data="https://github.com/EdwardZheng0312/3ddatadiver/blob/master/doc/3DdataDiver_poster.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/EdwardZheng0312/3ddatadiver/blob/master/doc/3DdataDiver_poster.pdf">
        <p>This browser does not support PDFs. To download the PDF, please view it: <a href="https://github.com/EdwardZheng0312/3ddatadiver/blob/master/doc/3DdataDiver_poster.pdf">Download PDF</a>.</p>
    </embed>
</object>
