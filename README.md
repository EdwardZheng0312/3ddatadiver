# 3DdataDiver

<img src="https://raw.githubusercontent.com/EdwardZheng0312/VisualanalysisAFM/master/doc/deep-diver.png" width="165" height="229">

3DdataDiver is a package for the processing and visualization of 3D AFM data.  Users interact with the package primarily through
a GUI that is locally run.  Upon uploading .mat or .HDF5 files hidden data cleaning functions are called that correct sample slope,
linearize Zsensor data, concat Zsensor with phase/amp data, and generate a three numpy arrays: full, approach, and retract dataset.
 
