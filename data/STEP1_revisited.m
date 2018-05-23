close all
clear all

filename = ('C:\Users\nako825\Desktop\Hydration Layers\FFM0012.h5'); % specify filename and directory
fileinfo = hdf5info(filename); % load info of .h5 file, this will help us access the variables stored inside

%%%%%%% access main data files as 3D matrices
AMPtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(1));
DRIVEtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(2));
PHASEtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(3));
RAWtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(4));
ZSNSRtot = hdf5read(fileinfo.GroupHierarchy.Groups(1).Datasets(5));

%%%%%%% access data for equivalent height (and other observable) images
%%%%%%% we do not really care about these variables as much, but they are available in case we ever need them
AMPim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(1));
HEIGHTim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(2));
PHASEim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(3));
ZSNSRim = hdf5read(fileinfo.GroupHierarchy.Groups(2).Datasets(4));

%%%%%%% access metadata
METAdata = hdf5read(fileinfo.GroupHierarchy.Attributes.Value.Data);
