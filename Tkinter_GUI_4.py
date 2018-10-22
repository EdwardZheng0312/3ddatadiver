import h5py
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import os
import tkinter as tk
import tkinter.messagebox as tkMessageBox

try:
    from tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk

    py3 = 0
except ImportError:
    import tkinter.ttk as ttk

    py3 = 1

Huge_Font = ("Vardana", 18)
Large_Font = ("Vardana", 14)
Small_Font = ("Vardana", 11)
Tiny_Font = ("Vardana", 8)
init = 0


class Sea(tk.Tk):
    """Control function for the windows translate """

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.tk.call('wm', 'iconphoto', self._w, PhotoImage(file='taiji.png'))  # Set up the iconphoto for our software
        tk.Tk.wm_title(self, "3ddatadiver")  # Set up the name of our software
        self.state('zoomed')  # Set up the inital window operation size

        container = tk.Frame(self)  # Define the properties of the bracket of windows inside the GUI
        container.pack(side="top", fill="both", expand=True)  # Define the layout of the bracket
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}  # Define the windows in the GUI
        for F in (data_cleaning, load_data, threeD_plot, twoD_slicing, animation, tutorial, acknowledge):
            frame = F(container, self)  # Call the certain window from the bracket

            self.frames[F] = frame  # Define each son windows in GUI
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(data_cleaning)  # Call the main window of the GUI

    def show_frame(self, cont):  # Show the call window
        frame = self.frames[cont]
        frame.tkraise()  # Raise the called window to the top


class data_cleaning(tk.Frame):
    """Data Preprocessing function to clean up the input data file and export the cleaned dataset"""
    def Curselect1(self, event):
        global controller
        widget = event.widget
        select = widget.curselection()
        controller = widget.get(select[0])
        return controller

    def get_source(self, source, export_filename, controller):
        """The function to return the inputs to the GUI functions"""
        global filename
        global export_filename0
        global FFM
        global file
        global Zsnsr
        global valu1
        global valu2
        global valu3
        global linearized
        global Phase_reduced_array_approach
        global Amp_reduced_array_approach
        global Phase_reduced_array_retract
        global Amp_reduced_array_retract
        global x_size
        global y_size
        filename = source.get()               # Export the input data file
        file = h5py.File(filename, "r+")
        FFM = file['FFM']
        Zsnsr = FFM['Zsnsr']
        valu1 = FFM['Phase']
        valu2 = FFM['Amp']
        valu3 = FFM['Drive']
        export_filename0 = export_filename.get()
        Phase_threeD_array = self.generatearray()[0]
        Amp_threeD_array = self.generatearray()[1]
        Drive_threeD_array = self.generatearray()[2]
        Zsnsr_threeD_array = self.generatearray()[3]
        arraytocorr = self.correct_slope(Zsnsr_threeD_array, Drive_threeD_array)[0]
        indZ = self.correct_slope(Zsnsr_threeD_array, Drive_threeD_array)[1]
        linearized = self.bin_array(arraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array)[0]
        Phase_reduced_array_approach = self.bin_array(arraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array)[3]
        Amp_reduced_array_approach = self.bin_array(arraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array)[4]
        Phase_reduced_array_retract = self.bin_array(arraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array)[5]
        Amp_reduced_array_retract = self.bin_array(arraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array)[6]
        x_size = len(Zsnsr[:,1,1])
        y_size = len(Zsnsr[1,:,1])
        return FFM, Zsnsr, valu1, valu2, valu3, export_filename0, Phase_threeD_array, Amp_threeD_array, Zsnsr_threeD_array, arraytocorr, indZ, linearized, Phase_reduced_array_approach, Amp_reduced_array_approach, Phase_reduced_array_retract, Amp_reduced_array_retract

    def generatearray(self):
        """Function to pull single dataset from FFM object and perform initial formatting.  The .h5 file must be opened,
        with the FFM group pulled out and the Zsnsr array generated before this function can be ran.
        must be run prior to generate_array.

        :param target: Name of single dataset given in list keys generated from load_h5 function.
        target parameter must be entered as a string.

        Output: formatted numpy array of a single dataset.

        Example:

        Phase = generate_array('Phase')

        print(Phase[3,3,3]) = 106.05377
        """
        #  Code is built for Fortran (column-major) formatted arrays and .h5 files/numpy default to row-major arrays.
        #  We need to transpose the data and then convert it to Fortran indexing (order = "F" command).
        temp1 = np.array(valu1)
        temp_1 = np.transpose(temp1)
        Phase_threeD_array = np.reshape(temp_1, (len(temp_1[:, 1, 1]), len(temp_1[1, :, 1]), len(temp_1[1, 1, :])), order="F")

        temp2 = np.array(valu2)
        temp_2 = np.transpose(temp2)
        Amp_threeD_array = np.reshape(temp_2, (len(temp_2[:, 1, 1]), len(temp_2[1, :, 1]), len(temp_2[1, 1, :])), order="F")

        temp3 = np.array(valu3)
        temp_3 = np.transpose(temp3)
        Drive_threeD_array = np.reshape(temp_3, (len(temp_3[:, 1, 1]), len(temp_3[1, :, 1]), len(temp_3[1, 1, :])), order="F")

        Zsnsr_temp1 = np.array(Zsnsr)
        Zsnsr_temp = np.transpose(Zsnsr_temp1)
        Zsnsr_threeD_array = np.reshape(Zsnsr_temp, (len(Zsnsr_temp[:, 1, 1]),
                                                     len(Zsnsr_temp[1, :, 1]), len(Zsnsr_temp[1, 1, :])), order="F")
        assert np.isfortran(Phase_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                            Needs to be column-major indexing."
        assert np.isfortran(Zsnsr_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                            Needs to be column-major indexing."
        assert len(temp1[1, 1, :]) == len(Phase_threeD_array[:, 1, 1]), "Transpose not properly applied, check \
                                                            dimensions of input array."
        assert len(Zsnsr_temp1[1, 1, :]) == len(Zsnsr_threeD_array[:, 1, 1]), "Transpose not properly applied, check \
                                                            dimensions of input array."
        return Phase_threeD_array, Amp_threeD_array, Drive_threeD_array, Zsnsr_threeD_array


    def correct_slope(self, Zsnsr_threeD_array, Drive_threeD_array):
        """Function that corrects for sample tilt and generates arrays used in the bin_array
        function.

        :param Zsnsr: 3D numpy array of a single signal feed.  For instance, the entire dataset of
        zsensor data.  The function is flexible enough to work on any raw data signal, but can
        only work on one at a time.

        Output: 3D numpy array with values adjusted for sample tilt, a 2D numpy array with
        the index of the max Zsensor value for each x,y coordinate, 1D numpy array of the X
        direction correction, and 1D numpy array of the Y direction correction.

        Example of function calling format:
        ZSNSRtotCORR, indZ, driftx, drifty = correct_slope(zsensor)"""
        global arraytotcorr
        global indZ

        assert np.isfortran(Zsnsr_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                        Needs to be column-major indexing."

        # Convert matrix from meters to nanometers.
        Zsnsr_threeD_array = np.multiply(Zsnsr_threeD_array, -1000000000)
        # We have create an numpy array of the correct shape to populate.
        array_min = np.zeros((len(Zsnsr_threeD_array[1, :, 1]), len(Zsnsr_threeD_array[1, 1, :])))
        indZ = np.zeros((len(Zsnsr_threeD_array[1, :, 1]), len(Zsnsr_threeD_array[1, 1, :])))
        # Populate zero arrays with min z values at all x,y positions.  Also, populate indZ array
        # with the index of the min z values for use in correct_Zsnsr()
        for j in range(len(Zsnsr_threeD_array[1, :, 1])):
            for i in range(len(Zsnsr_threeD_array[1, 1, :])):
                array_min[j, i] = (np.min(Zsnsr_threeD_array[:, i, j]))
                indZ[i, j] = np.min(np.where(Zsnsr_threeD_array[:, i, j] == np.min(Zsnsr_threeD_array[:, i, j])))

        # Find the difference between the max and mean values in the z-direction for
        # each x,y point. Populate new matrix with corrected values.
        Zdriftx = np.zeros(len(Zsnsr_threeD_array[1, :, 1]))
        Zdrifty = np.zeros(len(Zsnsr_threeD_array[1, 1, :]))
        corrected_array = np.zeros((len(Zsnsr_threeD_array[1, :, 1]), len(Zsnsr_threeD_array[1, 1, :])))

        # Correct the for sample tilt along to the y-direction, then correct for sample tilt
        # along the x-direction.
        for j in range(len(Zsnsr_threeD_array[1, :, 1])):
            for i in range(len(Zsnsr_threeD_array[1, 1, :])):
                Zdrifty[j] = np.mean(array_min[j, :])
                Zdriftx[i] = np.mean(corrected_array[:, i])

        ##################################################################################
        # We have create an numpy array of the correct shape to populate.
        drive_min = np.zeros((len(Drive_threeD_array[1, :, 1]), len(Drive_threeD_array[1, 1, :])))
        indD = np.zeros((len(Drive_threeD_array[1, :, 1]), len(Drive_threeD_array[1, 1, :])))
        # Populate zero arrays with min z values at all x,y positions.  Also, populate indZ array
        # with the index of the min z values for use in correct_Zsnsr()
        for j in range(len(Drive_threeD_array[1, :, 1])):
            for i in range(len(Drive_threeD_array[1, 1, :])):
                drive_min[j, i] = (np.min(Drive_threeD_array[:, i, j]))
                indD[i, j] = np.min(np.where(Drive_threeD_array[:, i, j] == np.min(Drive_threeD_array[:, i, j])))

        # Find the difference between the max and mean values in the z-direction for
        # each x,y point. Populate new matrix with corrected values.
        Ddriftx = np.zeros(len(Drive_threeD_array[1, :, 1]))
        Ddrifty = np.zeros(len(Drive_threeD_array[1, 1, :]))
        Dcorrected_array = np.zeros((len(Drive_threeD_array[1, :, 1]), len(Drive_threeD_array[1, 1, :])))

        # Correct the for sample tilt along to the y-direction, then correct for sample tilt
        # along the x-direction.
        for j in range(len(Drive_threeD_array[1, :, 1])):
            for i in range(len(Drive_threeD_array[1, 1, :])):
                Ddrifty[j] = np.mean(drive_min[j, :])
                Ddriftx[i] = np.mean(Dcorrected_array[:, i])

        ##################################################################################
        #CROP = 15
        #[zSIZE, xSIZE, ySIZE] = Zsnsr.shape
        #j1 = 1; j2 = ySIZE; i1 = 1; i2 = xSIZE

        #Zcorr = max(max(Zcorr[CROP : len(Zcorr) - CROP, CROP : len(Zcorr) - CROP])) - Zcorr
        #Dcorr = max(max(Dcorr[CROP : len(Dcorr) - CROP, CROP : len(Dcorr) - CROP])) - Dcorr
        #############################################################################################################

        # Apply corrected slope to each level of 3D numpy array
        arraytotcorr = np.empty_like(Zsnsr_threeD_array)
        for j in range(len(Zsnsr_threeD_array[1, :, 1])):
            for i in range(len(Zsnsr_threeD_array[1, 1, :])):
                arraytotcorr[:, i, j] = Zsnsr_threeD_array[:, i, j] - Zdriftx[i] - Zdrifty[j]

        assert (all((len(arraytotcorr)/2-200) <= value <= (len(arraytotcorr)/2+200) for value in indZ[1, :])) == True, \
            "Max extension in wrong location, check input array."
        assert (all((len(arraytotcorr)/2-200) <= value <= (len(arraytotcorr)/2+200) for value in indZ[:, 1])) == True, \
            "Max extension in wrong location, check input array."

        return arraytotcorr, indZ

    def bin_array(self, arraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array):
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

        # Generate empty numpy array to populate.
        global Phase_reduced_array
        global Amp_reduced_array
        global Phase_reduced_array_approach
        global Phase_reduced_array_retract
        global Amp_reduced_array_approach
        global Amp_reduced_array_retract
        global linearized

        assert np.isfortran(Phase_threeD_array) == True, "Input Phase array not passed through generate_array fucntion.  \
                                                        Needs to be column-major indexing."
        assert np.isfortran(Amp_threeD_array) == True, "Input Amp array not passed through generate_array fucntion.  \
                                                        Needs to be column-major indexing."

        arraymean = np.zeros(len(arraytotcorr[:, 1, 1]))
        # Create list of the mean Zsensor value for each horizontal slice of Zsensor array.
        for z in range(len(arraymean)):
            arraymean[z] = np.mean(arraytotcorr[z, :, :])
        # Turn mean Zsensor data into a linear vector with a step size of 0.02 nm.
        linearized = np.arange(-0.2, arraymean.max(), 0.02)
        # Generate empty array to populate
        Phase_reduced_array_approach = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        Amp_reduced_array_approach = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        Phase_reduced_array_retract = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        Amp_reduced_array_retract = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        # Cut raw phase/amp datasets into approach and retract, then bin data according to the linearized Zsensor data.
        # Generate new arrays from the means of each bin.  Perform on both approach and retract data.
        for j in range(len(arraytotcorr[1, :, 1])):
            for i in range(len(arraytotcorr[1, 1, :])):
                z = arraytotcorr[:(int(indZ[i, j])), i, j]  # Create dataset with just retract data
                digitized = np.digitize(z, linearized)  # Bin Z data based on standardized linearized vector.
                for n in range(len(linearized)):
                    ind = list(np.where(digitized == n)[0])  # Find which indices belong to which bins
                    Phase_reduced_array_approach[n, i, j] = np.mean(Phase_threeD_array[ind, i, j])  # Find the mean of Phase array's bins and
                                                                                    # populate new array.
                    Amp_reduced_array_approach[n, i, j] = np.mean(Amp_threeD_array[ind, i, j])    # Find the mean of Amp array's bins and
                                                                                    #  populate new array.

        for j in range(len(arraytotcorr[1, :, 1])):
            for i in range(len(arraytotcorr[1, 1, :])):
                z = arraytotcorr[-(int(indZ[i, j])):, i, j]  # Create dataset with just approach data.
                z = np.flipud(z)  # Flip array so surface is at the bottom on the plot.
                digitized = np.digitize(z, linearized)  # Bin Z data based on standardized linearized vector.
                for n in range(len(linearized)):
                    ind = list(np.where(digitized == n)[0])  # Find which indices belong to which bins
                    Phase_reduced_array_retract[n, i, j] = np.mean(Phase_threeD_array[ind, i, j])  # Find the mean of Phase array's bins and
                                                                                    # populate new array.
                    Amp_reduced_array_retract[n, i, j] = np.mean(Amp_threeD_array[ind, i, j])    # Find the mean of Amp array's bins and
                                                                                    #  populate new array.

        Phase_reduced_array = np.concatenate((Phase_reduced_array_approach, Phase_reduced_array_retract), axis=0)     #Merge Phase and Amp array into two different reduced array, contains approach and retract movement
        Amp_reduced_array =  np.concatenate((Amp_reduced_array_approach, Amp_reduced_array_retract), axis=0)

        return linearized, Phase_reduced_array, Amp_reduced_array, Phase_reduced_array_approach, Amp_reduced_array_approach, Phase_reduced_array_retract, Amp_reduced_array_retract

    def export_cleaned_data(self, file, Phase_reduced_array, Amp_reduced_array, export_filename0):
        global xSIZE
        global ySIZE
        global zSIZE
        global Xnm
        global Ynm

        h5file_approach = export_filename0 + str(".h5")  # Define the final name of the h5 file

        new_h5file = h5py.File(h5file_approach, 'w')     #Create the New H5 Files

        # Export the detailed information for user input HDF5 file attributes
        METAdata_convert = list(file.attrs.values())
        METAdata = str(METAdata_convert)

        string1 = METAdata.find('ThermalQ')
        string2 = METAdata.find('ThermalFrequency')

        Qfactor = METAdata[string1 + len(str('ThermalQ')) + 1: string2 - 2]

        string3 = METAdata.find('ThermalWhiteNoise')

        FreqRes = METAdata[string2 + len(str('ThermalFrequency')) + 1: string3 - 2]

        string4 = METAdata.find('DriveAmplitude')
        string5 = METAdata.find('DriveFrequency')

        AmpDrive = METAdata[string4 + len(str('DriveAmplitude')) + 1: string5 - 2]

        string6 = METAdata.find('AmpInvOLS')
        string7 = METAdata.find('UpdateCounter')

        AmpInvOLS = METAdata[string6 + len(str('AmpInvOLS')) + 1: string7 - 2]

        string8 = METAdata.find('DriveFrequency')
        string9 = METAdata.find('SweepWidth')

        FreqDrive = METAdata[string8 + len(str('DriveFrequency')) + 1: string9 - 2]

        string10 = METAdata.find('Initial FastScanSize:')
        string11 = METAdata.find('Initial SlowScanSize:')

        Xnm = METAdata[string10 + len(str('Initial FastScanSize:')) + 1: string11 - 2]

        string12 = METAdata.find('Initial ScanRate:')

        Ynm = METAdata[string11 + len(str('Initial SlowScanSize:')) + 1: string12 - 2]

        new_h5file_g1 = new_h5file.create_group('important_data')
        new_h5file_g2 = new_h5file.create_group('nonimportant_data')
        new_h5file_g3 = new_h5file.create_group('export_parameters')

        new_h5file_g1.create_dataset('PHASEphaseD', data = Phase_reduced_array, dtype='f4')
        new_h5file_g1.create_dataset('AMPampD', data = Amp_reduced_array, dtype='f4')
        new_h5file_g1.create_dataset('Dlinear', data=Amp_reduced_array, dtype='f4')
        new_h5file_g1.create_dataset('Dcorr', data=Amp_reduced_array, dtype='f4')

        new_h5file_g2.create_dataset('Ddriftx', data=Amp_reduced_array, dtype='f4')
        new_h5file_g2.create_dataset('Ddrifty', data=Amp_reduced_array, dtype='f4')
        new_h5file_g2.create_dataset('Zbin', data=Amp_reduced_array, dtype='f4')
        new_h5file_g2.create_dataset('CROP', data=Amp_reduced_array, dtype='f4')

        attrs_export = np.string_(dict([("AmpInvOLS", AmpInvOLS), ("AmpDrive", AmpDrive), ("Qfactor", Qfactor), ("FreqDrive", FreqDrive), ("FreqRes", FreqRes), ("Xnm", Xnm), ("Ynm", Ynm)]))

        new_h5file_g3.create_dataset('METAdata', data=METAdata_convert)
        new_h5file_g3.create_dataset('Attrs_info_input_HDF5', data=attrs_export)
        return new_h5file, Xnm, Ynm

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')

        label1 = ttk.Label(self, text="Step 1: Data Pre-Processing", font=Huge_Font, background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Input Filename", font=Large_Font, background='#ffffff')
        label2.pack()
        source = ttk.Entry(self)
        source.pack(pady=10, padx=10)

        label3 = ttk.Label(self, text='Export Clean Dataset Name', font=Large_Font, background='#ffffff')
        label3.pack(pady=10, padx=10)
        export_filename = ttk.Entry(self)
        export_filename.pack()

        label4 = ttk.Label(self, text="Slope Correction Switch", font=Large_Font, background='#ffffff')
        label4.pack(padx=10, pady=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, 'On')
        listbox.insert(2, 'Off')
        listbox.bind('<<ListboxSelect>>', self.Curselect1)

        button0 = ttk.Button(self, text="Load File",command=lambda: self.get_source(source, export_filename, controller))
        button0.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Export as HDF5 File",command=lambda: self.export_cleaned_data(file, Phase_reduced_array, Amp_reduced_array, export_filename0))
        button1.pack(pady=5, padx=5)

        button2 = ttk.Button(self, text="Plot Data", command=lambda: controller.show_frame(load_data))
        button2.pack(pady=5, padx=5)

        button3 = ttk.Button(self, text="Tutorial", command=lambda: controller.show_frame(tutorial))
        button3.pack(pady=5, padx=5)

        button4 = ttk.Button(self, text="Acknowledgements", command=lambda: controller.show_frame(acknowledge))
        button4.pack(pady=5, padx=5)

        button5 = ttk.Button(self, text="Quit", command=lambda: controller.quit())
        button5.pack(pady=5, padx=5)


class load_data(tk.Frame):
    """The function for user to input the objectives for further visualization interests based"""
    def Curselect2(self, event):
        """The mouse click event for selecting the objectives you are interested to cleanup"""
        global valu
        widget = event.widget  # Define the event of the objective from the GUI
        select = widget.curselection()  # Read the selection from the GUI objectives
        valu = widget.get(select[0])    # Return the selection from the GUI objectives
        return valu

    def get_data(self, Xnm, Ynm, Phase_reduced_array_retract, Phase_reduced_array_approach, Amp_reduced_array_approach, Amp_reduced_array_retract):
        """The function to return the inputs to the GUI functions"""
        global x_actual
        global y_actual
        global reduced_array_retract
        global reduced_array_approach

        Xnm = (np.fromstring(Xnm, dtype=float, sep=' ') * 1e9)
        Ynm = (np.fromstring(Ynm, dtype=float, sep=' ') * 1e9)

        x_actual = int(Xnm)  # Export the actual size of x direction from the AFM measurements
        y_actual = int(Ynm)  # Export the actual size of y direction from the AFM measurements

        if valu == 'Phase':
            reduced_array_retract = Phase_reduced_array_retract
            reduced_array_approach = Phase_reduced_array_approach
        else:
            reduced_array_retract = Amp_reduced_array_retract
            reduced_array_approach = Amp_reduced_array_approach
        return x_actual, y_actual, reduced_array_retract, reduced_array_approach


    def __init__(self, parent, controller):  # Define all the controller in the load_data window
        global txtxactual
        global txtyactual
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 2: Input Dataset Information for Visualization", font=Huge_Font,
                          background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Select the Objectives", font=Large_Font, background='#ffffff')
        label2.pack(padx=10, pady=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, 'Amp')
        listbox.insert(2, 'Phase')
        listbox.bind('<<ListboxSelect>>', self.Curselect2)

        button0 = ttk.Button(self, text="Apply Parameters",command=lambda: self.get_data(Xnm, Ynm, Phase_reduced_array_retract, Phase_reduced_array_approach, Amp_reduced_array_retract, Amp_reduced_array_approach))
        button0.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="3D Plot", command=lambda: controller.show_frame(threeD_plot))
        button1.pack(pady=5, padx=5)

        button2 = ttk.Button(self, text="2D Slicing", command=lambda: controller.show_frame(twoD_slicing))
        button2.pack(pady=5, padx=5)

        button3 = ttk.Button(self, text="2D Slicing Animation", command=lambda: controller.show_frame(animation))
        button3.pack(pady=5, padx=5)

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: (self.clean(), self.update_idletasks()))
        button4.pack(pady=5, padx=5)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=5, padx=5)


class threeD_plot(tk.Frame):
    """The function for the 3D plot"""

    def Curselect3(self, event):
        """Export the Z_direction information for the AFM cantilever motion from the GUI"""
        global Z_direction
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        return Z_direction

    def clear(self):
        canvas.get_tk_widget().destroy()  # Clean up the export the figure in window

    def threeDplot(self, Z_direction, x_actual, y_actual):
        """3D plot function"""
        global canvas
        if Z_direction == "Up":                         # If the AFM cantilever moves upward and return the z axis information and the corresponding the data points in that direction and also redefine the index and the sequence of the data points
            Z_dir = np.flip(linearized, axis=0)
            data1 = reduced_array_retract
        else:                                           # If the AFM cantilever moves downward and return the z axis information and the corresponding the data points in that direction and also redefine the index and the sequence of the data points
            Z_dir = linearized
            data1 = reduced_array_approach

        data1[np.isnan(data1)] = np.nanmin(data1)  # Replace NaN with min value of array.

        x = np.linspace(init, x_actual, len(data1[1,:,1]))         # Define the plotting valuable x
        y = np.linspace(init, y_actual, len(data1[1,1,:]))         # Define the plotting valuable y
        z = np.linspace(init, Z_dir.max(), len(data1[:,1,1]))      # Define the plotting valuable z

        xi,zi,yi = np.meshgrid(x,z,y)

        fig = plt.figure(figsize=(11, 9), facecolor='white')       #Define the figure to make a plot
        ax = fig.add_subplot(111, projection='3d')                 #Define the 3d plot
        # Define the scatter plot
        im = ax.scatter(xi, yi, zi, c=data1.flatten(), vmax=np.nanmax(data1), vmin=np.nanmin(data1))
        plt.colorbar(im)                                           # Define the colorbar in the scatter plot
        ax.set_xlim(left=init, right=x_actual)                     # Define the X limit for the plot
        ax.set_ylim(top=y_actual, bottom=init)                     # Define the Y limit for the plot
        ax.set_zlim(top=np.nanmax(Z_dir), bottom=init)             # Define the Z limit for the plot
        ax.set_xlabel('X(nm)', fontsize=15)                        # Define the X label for the plot
        ax.set_ylabel('Y(nm)', fontsize=15)                        # Define the Y label for the plot
        ax.set_zlabel('Z(nm)', fontsize=15)                        # Define the Z label for the plot
        # Define the title for the plot
        ax.set_title('3D Plot for _' + str(Z_direction) + '_' + str(valu) + ' of the AFM data', fontsize=20, y=1.05)

        canvas = FigureCanvasTkAgg(fig, self)                      # Define the display figure in the window
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP)                   # Define the display region in GUI

        fig.savefig("3D Plot_" + str(Z_direction)+ str(valu) + ".tif")  # Save the export figure as tif file

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 3: 3D Plot", font=Huge_Font, background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label2.pack(pady=10, padx=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Up")
        listbox.insert(2, "Down")
        listbox.bind('<<ListboxSelect>>', self.Curselect3)

        button1 = ttk.Button(self, text="Get 3D Plot", command=lambda: self.threeDplot(Z_direction, x_actual, y_actual))
        button1.pack(pady=5, padx=5)

        button2 = ttk.Button(self, text="Clear the Inputs", command=lambda: self.clear())
        button2.pack(pady=5, padx=5)

        button3 = ttk.Button(self, text="2D Slicing", command=lambda: controller.show_frame(twoD_slicing))
        button3.pack(pady=5, padx=5)

        button4 = ttk.Button(self, text="Back to Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack(pady=5, padx=5)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=5, padx=5)


class twoD_slicing(tk.Frame):
    """The functions for different scopes of 2D slicings"""

    def export_filename(self, txtfilename):
        """Export the user input export filename into the GUI"""
        global export_filename2
        export_filename2 = txtfilename.get()  # Return the name of export file
        return export_filename2

    def Curselect4(self, event):
        global Z_direction
        global Z_dir
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        if Z_direction == "Up":
            Z_dir = np.flip(linearized, axis=0)
            print(len(linearized))
        else:
            Z_dir = linearized
        return Z_dir

    def location_slices(self, txtnslices):
        """Define the location of the slice you are interested"""
        global location_slices
        location_slices = round(float(txtnslices.get()), 2)  # Export the locations of slice from the GUI
        return location_slices

    def pixel_converter(self, location_slices):
        """Convert from the real z to pixel"""
        global location_slices_pixel_x
        global location_slices_pixel_y
        global location_slices_pixel_z
        location_slices_pixel_x = int(round(float(location_slices / x_actual), 2) * (x_size-1))
        location_slices_pixel_y = int(round(float(location_slices / y_actual), 2) * (y_size-1))
        location_slices_pixel_z = int(float(location_slices / round(float(np.array(Z_dir).max()),2)) * len(Z_dir)) + 1
        return location_slices_pixel_x, location_slices_pixel_y, location_slices_pixel_z

    def clear(self):
        """Clear up the inputs"""
        txtnslices.delete(0, END)
        txtfilename.delete(0, END)
        canvas1.get_tk_widget().destroy()
        canvas2.get_tk_widget().destroy()

    def create_pslist(self, Z_direction):
        """The function for reshape the input data file depends on certain shape of the input data file, and also judge
         the AFM cantilever movement direction"""
        global pslist
        if Z_direction == "Up":
            pslist = reduced_array_retract
            print(np.shape(reduced_array_retract))
        else:
            pslist = reduced_array_approach
        return pslist

    def twoDX_slicings(self, location_slices_pixel_x, export_filename2, x_actual, y_actual):
        """Plotting function for the X direction slicing"""
        global canvas1
        global canvas2
        if location_slices_pixel_x in range(x_size + 1):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for X is between 0 to " + str(x_actual) + ".")

        As = np.array(self.create_pslist(Z_direction))[:, location_slices_pixel_x, :]  # Select the phaseshift data points in the certain slice plane

        a = np.linspace(init, x_actual, x_size)[location_slices_pixel_x]  # Define the certain x slice in the x space
        print(location_slices_pixel_x)
        b = np.linspace(init, y_actual, x_size)  # Define the y space
        c = Z_dir  # Define the z space
        X, Z, Y = np.meshgrid(a, c, b)  # Create the meshgrid for the 3d space

        fig = plt.figure(figsize=(9, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        # Define the fixed colorbar range based the overall phaseshift values from the input data file
        im = ax.scatter(X, Y, Z, c=As.flatten(), s=6, vmax=np.array(self.create_pslist(Z_direction)).max(),
                        vmin=np.array(self.create_pslist(Z_direction)).min())
        #Add colorbar
        cbar = plt.colorbar(im)
        # Label the colorbar
        cbar.set_label(str(valu), rotation=90)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D X Slicing (X=' + str(round(a, 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)

        canvas1 = FigureCanvasTkAgg(fig, self)  # Plot the 3D figure of 2D slicing
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h5file = export_filename2 + str(Z_direction) + str("_X.h5")  # Define the final name of the h5 file

        # Assuming As is a list of lists
        h = h5py.File(h5file, 'w')  # Create the empty h5 file
        h.create_dataset("data", data=As)  # Insert the data into the empty file

        setStr = '{}_Xslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig1 = plt.figure(figsize=(11, 9), facecolor='white')
        plt.subplot(111)
        plt.imshow(As, aspect='auto', origin="lower", vmax=np.array(self.create_pslist(Z_direction)).max(),
                   vmin=np.array(self.create_pslist(Z_direction)).min())
        plt.axis([init, y_size - 1, init, len(Z_dir) - 1])  # Adjust the axis range for the 2D slicing
        plt.xlabel('Y', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D X Slicing (X=' + str(round(a, 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        #  Add and label colorbar
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig1, self)  # Plot the 2D figure of 2D slicing
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Xslices.tif'.format(export_filename2)  # Define the export image name
        fig1.savefig(setStr)

    def twoDY_slicings(self, location_slices_pixel_y, export_filename2, x_actual, y_actual):
        """Plotting function for the Y direction slicing"""
        global canvas1
        global canvas2
        if location_slices_pixel_y in range(y_size + 1):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for Y is between 0 to " + str(y_actual) + ".")
        a = np.linspace(init, x_actual, x_size)
        print(location_slices_pixel_y)
        b = np.linspace(init, y_actual, y_size)[location_slices_pixel_y]
        c = Z_dir
        X, Z, Y = np.meshgrid(a, c, b)

        Bs = np.array(self.create_pslist(Z_direction))[:, location_slices_pixel_y, :]

        fig = plt.figure(figsize=(9, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=Bs.flatten())
        #  Add and lable colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(str(valu))
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D Y Slicing (Y=' + str(round(b, 3)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)

        canvas1 = FigureCanvasTkAgg(fig, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h5file = export_filename2 + str(Z_direction) + str("_Y.h5")

        # Assuming Bs is a list of lists
        h = h5py.File(h5file, 'w')
        h.create_dataset("data", data=Bs)

        setStr = '{}_Yslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig2 = plt.figure(figsize=(11, 9), facecolor='white')
        plt.subplot(111)
        plt.imshow(Bs, aspect='auto', vmax=np.array(self.create_pslist(Z_direction)).max(),
                   vmin=np.array(self.create_pslist(Z_direction)).min())
        plt.axis([init, x_size- 1, init, len(Z_dir) - 1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D Y Slicing (Y=' + str(round(b, 3)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        #  Add and label colorbar
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig2, self)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Yslices.tif'.format(export_filename2)
        fig2.savefig(setStr)

    def twoDZ_slicings(self, location_slices_pixel_z, export_filename2, x_actual, y_actual):
        """3D Plotting function for Z direction slicing"""
        global canvas1
        if location_slices_pixel_z in range (len(Z_dir) + 2):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for Z is between 0 to " + str(np.array(Z_dir).max()) + ".")

        phaseshift = (self.create_pslist(Z_direction))[location_slices_pixel_z]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        print(location_slices_pixel_z)
        X, Z, Y = np.meshgrid(a, Z_dir[location_slices_pixel_z], b)
        l = phaseshift

        fig = plt.figure(figsize=(9, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=l.flatten(), s=6, vmax=np.array(self.create_pslist(Z_direction)).max(), vmin=np.array(self.create_pslist(Z_direction)).min())

        #  Add and label colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(str(valu))
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D Z Slicing (Z=' + str(round(Z_dir[(location_slices_pixel_z) - 1], 4)) + 'nm) for the ' + str(
            valu) + ' of AFM data', fontsize=13)

        canvas1 = FigureCanvasTkAgg(fig, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h5file = export_filename2 + str(Z_direction) + str("_Z.h5")

        # Assuming phaseshift is a list of lists
        h = h5py.File(h5file, 'w')
        h.create_dataset("data", data=phaseshift)

        setStr = '{}_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)

    def twoZ_slicings(self, location_slices_pixel_z, export_filename2, x_actual, y_actual):
        """2D Plotting function for Z direction slicing"""
        global canvas2
        phaseshift = (self.create_pslist(Z_direction))[location_slices_pixel_z - 1]

        l = phaseshift

        fig = plt.figure(figsize=(9, 9), facecolor='white')
        plt.imshow(l, vmax=np.array(self.create_pslist(Z_direction)).max(),
                   vmin=np.array(self.create_pslist(Z_direction)).min())
        plt.axis([init, x_size - 1, init, y_size - 1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z=' + str(round(Z_dir[(location_slices_pixel_z) - 1], 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        #  Add and label colorbar
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig, self)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig1 = plt.figure(figsize=(9, 9), facecolor='white')
        plt.imshow(l, vmax=np.array(self.create_pslist(Z_direction)).max(),
                   vmin=np.array(self.create_pslist(Z_direction)).min())
        plt.axis([init, x_size - 1, init, y_size - 1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z=' + str(round(Z_dir[(location_slices_pixel_z) - 1], 4)) + 'nm) for the '
                  + str(valu) + ' of AFM data', fontsize=13)
        plt.colorbar()
        x = np.array(plt.ginput(1))
        canvas3 = FigureCanvasTkAgg(fig1, self)
        canvas3.show()

    def __init__(self, parent, controller):
        global txtnslices
        global txtzdir
        global txtfilename
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 4: 2D Slicing", font='Huge_Font', background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Slices Location (nm)", font="Small_Font", background='#ffffff')
        label2.pack(pady=10, padx=10)
        txtnslices = ttk.Entry(self)
        txtnslices.pack()

        label3 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label3.pack(pady=10, padx=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Up")
        listbox.insert(2, "Down")
        listbox.bind('<<ListboxSelect>>', self.Curselect4)

        label4 = ttk.Label(self, text="Export Filename", font="Small_Font", background='#ffffff')
        label4.pack(pady=10, padx=10)
        txtfilename = ttk.Entry(self)
        txtfilename.pack()

        button0 = ttk.Button(self, text="Get Location Slices & Directions",
                             command=lambda: (self.location_slices(txtnslices), self.export_filename(txtfilename),
                                              self.pixel_converter(location_slices)))
        button0.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Get 2D X Slicing Plot",
                             command=lambda: self.twoDX_slicings(location_slices_pixel_x, export_filename2,
                                                                 x_actual, y_actual))
        button1.pack(pady=5, padx=5)

        button2 = ttk.Button(self, text="Get 2D Y Slicing Plot",
                             command=lambda: self.twoDY_slicings(location_slices_pixel_y, export_filename2,
                                                                 x_actual, y_actual))
        button2.pack(pady=5, padx=5)

        button3 = ttk.Button(self, text="Get 2D Z Slicing Plot", command=lambda:
        (self.twoDZ_slicings(location_slices_pixel_z, export_filename2, x_actual, y_actual,
                             ), self.twoZ_slicings(location_slices_pixel_z, export_filename2, x_actual,
                                                                 y_actual)))
        button3.pack(pady=5, padx=5)

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: (self.clear()))
        button4.pack(pady=5, padx=5)

        button5 = ttk.Button(self, text="3D Plot", command=lambda: controller.show_frame(threeD_plot))
        button5.pack(pady=5, padx=5)

        button6 = ttk.Button(self, text="2D Slicing Animation", command=lambda: controller.show_frame(animation))
        button6.pack(pady=5, padx=5)

        button7 = ttk.Button(self, text="Back to Organize Dataset", command=lambda: controller.show_frame(load_data))
        button7.pack(pady=5, padx=5)

        button8 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button8.pack(pady=5, padx=5)

        label5 = ttk.Label(self, text="The reference level for the plots is set as zero at the substrate surface.",
                           font=(None, 10))
        label5.pack()


class animation(tk.Frame):
    global event1

    def Curselect5(self, event):
        global Z_direction
        global Z_dir
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        if Z_direction == "Up":
            Z_dir = np.flip(linearized, axis=0)
        else:
            Z_dir = linearized
        return Z_dir

    def create_pslist(self, Z_direction):
        """The function for reshape the input data file depends on certain shape of the input data file, and also judge
         the AFM cantilever movement direction"""
        global pslist
        if Z_direction == "Up":
            pslist = reduced_array_retract
        else:
            pslist = reduced_array_approach
        return pslist

    def slice(self, numslices):
        global num_slice
        num_slice = int(numslices.get())
        return num_slice

    def Curselect5(self, event):  # Slicing Directions
        global Dir
        widget = event.widget
        select = widget.curselection()
        Dir = widget.get(select[0])
        return Dir

    def save_Z_animation(self, Z_dir, x_actual, y_actual, x_size, y_size,
                         pslist):  # x_num_slice, y_num_slice, z_num_slice
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('Slicing Animation for the  AFM Phase Shift', fontsize=20)
        # ----------------------------------------------------------------------------------------------------------------
        ims = []
        for add in range(16):
            a = np.linspace(init, x_actual, x_size)
            b = np.linspace(init, y_actual, 64)
            c = Z_dir[(14 * add)]
            x, z, y = np.meshgrid(a, c, b)
            l = np.array(pslist)[(230 - 14 * add), :, :]
            plane = ax.scatter(x, y, z, c=l.flatten(), s=6)
            # fig.colorbar(plane, fraction=0.026, pad=0.04)
            ims.append((plane,))  # XY slice
        for add in range(16):
            a = np.linspace(init, x_actual, x_size)
            b = np.linspace(init, y_actual, 64)[4 * add]
            c = Z_dir
            x, z, y = np.meshgrid(a, c, b)
            l = np.array(pslist)[init:len(Z_dir), :, 4 * add]
            ims.append((ax.scatter(x, y, z, c=l.flatten(), s=6),))  # XZ slice
        for add in np.arange(16):
            a = np.linspace(init, x_actual, x_size)[4 * add]
            b = np.linspace(init, y_actual, 64)
            c = Z_dir
            x, z, y = np.meshgrid(a, c, b)
            l = np.array(pslist)[init:len(Z_dir), 4 * add, init:y_size]
            ims.append((ax.scatter(x, y, z, c=l.flatten(), s=6),))  # YZ slice
        im_ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=12000, blit=True)  # 之前interval是500

        plt.rcParams['animation.ffmpeg_path'] = 'D:\ImageMagick-7.0.8-Q16\ffmpeg.exe'
        plt.rcParams["animation.convert_path"] = "D:\ImageMagick-7.0.8-Q16\magick.exe"
        im_ani.save('XYZ Slicing animation.gif', writer='imagemagick', extra_args="convert", fps=300)
        return

    def __init__(self, parent, controller):
        global numslices
        global txtfilename2
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Step 5: 2D Slicing Animation", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label1 = ttk.Label(self, text="Number of Slices", font='Large_Font', background='#ffffff')
        label1.pack()
        numslices = ttk.Entry(self)
        numslices.pack()

        label2 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label2.pack(pady=10, padx=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Up")
        listbox.insert(2, "Down")
        listbox.bind('<<ListboxSelect>>', self.Curselect5)

        button2 = ttk.Button(self, text="Save Animation",
                             command=lambda: self.save_Z_animation(Z_dir, x_actual, y_actual, x_size, y_size, pslist))
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button4.pack(pady=10, padx=10)


class tutorial(ttk.Frame):
    """The function for making the tutorial about this GUI"""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Tutorial", font=Large_Font, background='#ffffff')
        label1.pack()

        label2 = ttk.Label(self, text="Introduction to Our Software", font=Large_Font, background='#ffffff')
        label2.pack()

        def source():
            """Export the video for this GUI"""
            os.system("D:/New/Dropbox/UW/training/Cleanroom/EPFMNMEM2016-V004900_DTH.mp4")

        vid = ttk.Button(self, text="Play Video", command=source)
        vid.pack(pady=10, padx=10)

        button0 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button0.pack()


class acknowledge(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Acknowledge", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label_1 = ttk.Label(self,
                            text="This software is supported by the Pacific Northwest National Lab and the"
                                 " DIRECT Program in University of Washington.",
                            background='#ffffff', font='Small_Font')
        label_1.pack()

        photo1 = PhotoImage(file="PNNL.png")
        photo2 = PhotoImage(file="UWDIRECT.png")
        img1 = tk.Label(self, image=photo1, background='#ffffff')
        img2 = tk.Label(self, image=photo2, background='#ffffff')
        img1.image = photo1
        img2.image = photo2
        img1.pack()
        img2.pack()

        label_2 = ttk.Label(self, text="The software is created by Ellen Murphy, Xueqiao Zhang, Renlong Zheng.",
                            font='Small_Font', background='#ffffff')
        label_2.pack(pady=40, padx=10)

        button1 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button1.pack()


app = Sea()
app.mainloop()
