import h5py
import itertools
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import mpldatacursor
import numpy as np
import os
from statistics import mean
import tkinter as tk
from tkinter.filedialog import askopenfilename
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
    """Control function to build the environment and rendering of the GUI. Background function, user does not
    interact."""
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Set up the iconphoto for the GUI
        self.tk.call('wm', 'iconphoto', self._w,
                     PhotoImage(file=os.path.join('D:/1UW/3ddatadiver/3ddatadiver','taiji.png')))
        tk.Tk.wm_title(self, "3ddatadiver")  # Set up the name of our software
        self.state('zoomed')  # Set up the initial window operation size

        container = tk.Frame(self)  # Define the properties of the bracket of windows inside the GUI
        container.pack(side="top", fill="both", expand=True)  # Define the layout of the bracket
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}  # Define the windows in the GUI
        for F in (data_cleaning, load_data, Force_Curve_plot,
                  threeD_plot, twoD_slicing, animation_cool, tutorial, acknowledge):
            frame = F(container, self)  # Call the certain window from the bracket

            self.frames[F] = frame  # Define each son windows in GUI
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(data_cleaning)  # Call the main window of the GUI

    def show_frame(self, cont):  # Show the call window
        frame = self.frames[cont]
        frame.tkraise()  # Raise the called window to the top


class data_cleaning(tk.Frame):
    """Data Preprocessing function to clean up the input data file and export the cleaned datasets"""

    # Set up buttons.
    def Curselect1(self, event):
        global controller1
        widget = event.widget
        select = widget.curselection()
        controller1 = widget.get(select[0])
        return controller1

    def Curselect2(self, event):
        global controller2
        widget = event.widget
        select = widget.curselection()
        controller2 = widget.get(select[0])
        return controller2

    def openfilename(self):
        global fileName
        fileName = askopenfilename()
        return fileName

    def get_source(self, export_filename, controller2):
        """This function mines the dataset to generate objects and stores them for use in the GUI."""
        global filename
        global export_filename0
        global FFM
        global file
        global Zsnsr
        global valu1
        global valu2
        global valu3
        global valu4
        global Zbin
        global Zlinearized
        global Dlinearized
        global Zarraytotcorr
        global Darraytotcorr
        global Zcorrected_array
        global Dcorrected_array
        global Ddriftx
        global Ddrifty
        global indZ
        global indD
        global Phase_reduced_array
        global Amp_reduced_array
        global Phase_reduced_array_approach
        global Phase_reduced_array_retract
        global Amp_reduced_array_approach
        global Amp_reduced_array_retract
        global Phase_reduced_array_D
        global Amp_reduced_array_D
        global Phase_reduced_array_approach_D
        global Phase_reduced_array_retract_D
        global Amp_reduced_array_approach_D
        global Amp_reduced_array_retract_D
        global x_size
        global y_size
        file = h5py.File(fileName, "r+")
        FFM = file['FFM']
        Zsnsr = FFM['Raw']
        valu1 = FFM['Phase']
        valu2 = FFM['Amp']
        valu3 = FFM['Drive']
        export_filename0 = export_filename.get()
        Zbin = float(zbinsize.get())

        #  This is where the user can tell the software if the tip deflection data was collected on the input file.
        if controller2 == 'On':
            valu4 = FFM['Defl']
        else:
            valu4 = 'NaN'

        # Generate arrays from h5 file.
        Phase_threeD_array, Amp_threeD_array, Drive_threeD_array, Zsnsr_threeD_array = self.generatearray()

        # Correct Slope
        Zdriftx, Zdrifty, Zcorrected_array, Zarraytotcorr, indZ, CROP = self.correct_slope(Zsnsr_threeD_array,
                                                                                           controller1)
        Ddriftx, Ddrifty, Dcorrected_array, Darraytotcorr, indD, CROP = self.correct_slope(Drive_threeD_array,
                                                                                           controller1)
        # Bin array
        Zlinearized, Phase_reduced_array, Phase_reduced_array_approach, Phase_reduced_array_retract, Amp_reduced_array,\
            Amp_reduced_array_approach, Amp_reduced_array_retract  = \
            self.bin_array(Zarraytotcorr, indZ, Phase_threeD_array, Amp_threeD_array)
        Dlinearized, Phase_reduced_array_D, Phase_reduced_array_approach_D, Phase_reduced_array_retract_D, \
            Amp_reduced_array_D, Amp_reduced_array_approach_D, Amp_reduced_array_retract_D = \
            self.bin_array(Darraytotcorr, indD, Phase_threeD_array, Amp_threeD_array)
        x_size = len(Zsnsr[:,1,1])
        y_size = len(Zsnsr[1,:,1])
        return FFM, Zsnsr, valu1, valu2, valu3, valu4, Zbin, export_filename0, Phase_threeD_array, Amp_threeD_array, \
               Zsnsr_threeD_array, Darraytotcorr, Zarraytotcorr, indZ, indD, Zlinearized, Dlinearized, \
               Phase_reduced_array_approach, Amp_reduced_array_approach, Phase_reduced_array_retract, \
               Amp_reduced_array_retract, Phase_reduced_array_approach_D,  Amp_reduced_array_approach_D, \
               Phase_reduced_array_retract_D, \
               Amp_reduced_array_retract_D, Phase_reduced_array, Amp_reduced_array

    def export_HDF5(self):
        new_h5file, Xnm, Ynm = self.export_cleaned_data(file, Ddriftx, Ddrifty, Phase_reduced_array_retract_D, Amp_reduced_array_retract_D, valu4, Dcorrected_array, Dlinearized, export_filename0, CROP, controller2)
        return new_h5file, Xnm, Ynm

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
        Phase_threeD_array = np.reshape(temp_1, (len(temp_1[:, 1, 1]), len(temp_1[1, :, 1]),
                                                 len(temp_1[1, 1, :])), order="F")

        temp2 = np.array(valu2)
        temp_2 = np.transpose(temp2)
        Amp_threeD_array = np.reshape(temp_2, (len(temp_2[:, 1, 1]), len(temp_2[1, :, 1]),
                                               len(temp_2[1, 1, :])), order="F")

        temp3 = np.array(valu3)
        temp_3 = np.transpose(temp3)
        Drive_threeD_array = np.reshape(temp_3, (len(temp_3[:, 1, 1]), len(temp_3[1, :, 1]),
                                                 len(temp_3[1, 1, :])), order="F")

        Zsnsr_temp1 = np.array(Zsnsr)
        Zsnsr_temp = np.transpose(Zsnsr_temp1)
        Zsnsr_threeD_array = np.reshape(Zsnsr_temp, (len(Zsnsr_temp[:, 1, 1]), len(Zsnsr_temp[1, :, 1]),
                                                     len(Zsnsr_temp[1, 1, :])), order="F")

        ### Unit tests for generatearray function

        assert np.isfortran(Phase_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                            Needs to be column-major indexing."
        assert np.isfortran(Amp_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                                    Needs to be column-major indexing."
        assert np.isfortran(Drive_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                                    Needs to be column-major indexing."
        assert np.isfortran(Zsnsr_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                            Needs to be column-major indexing."
        assert len(temp1[1, 1, :]) == len(Phase_threeD_array[:, 1, 1]), "Transpose not properly applied, check \
                                                            dimensions of input array."
        assert len(temp2[1, 1, :]) == len(Amp_threeD_array[:, 1, 1]), "Transpose not properly applied, check \
                                                                    dimensions of input array."
        assert len(temp3[1, 1, :]) == len(Drive_threeD_array[:, 1, 1]), "Transpose not properly applied, check \
                                                                    dimensions of input array."
        assert len(Zsnsr_temp1[1, 1, :]) == len(Zsnsr_threeD_array[:, 1, 1]), "Transpose not properly applied, check \
                                                            dimensions of input array."
        return Phase_threeD_array, Amp_threeD_array, Drive_threeD_array, Zsnsr_threeD_array

    def correct_slope(self, target_threeD_array, controller1):
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

        assert np.isfortran(target_threeD_array) == True, "Input array not passed through generate_array fucntion.  \
                                                        Needs to be column-major indexing."
        global CROP

        # Convert Zsnsr matrix from meters to nanometers.
        target_threeD_array = np.multiply(target_threeD_array, -1000000000)

        CROP = 15
        [zSIZE, xSIZE, ySIZE] = np.shape(target_threeD_array)

        # We have create an numpy array of the correct shape to populate.
        array_min = np.zeros((len(target_threeD_array[1, :, 1]), len(target_threeD_array[1, 1, :])))
        ind = np.zeros((len(target_threeD_array[1, :, 1]), len(target_threeD_array[1, 1, :])))
        # Populate zero arrays with min z values at all x,y positions.  Also, populate indZ array
        # with the index of the min z values for use in correct_Zsnsr()
        for j in range(len(target_threeD_array[1, :, 1])):
            for i in range(len(target_threeD_array[1, 1, :])):
                array_min[j, i] = (np.min(target_threeD_array[:, i, j]))
                ind[i, j] = np.min(np.where(target_threeD_array[:, i, j] == np.min(target_threeD_array[:, i, j])))

        # Find the difference between the max and mean values in the z-direction for
        # each x,y point. Populate new matrix with corrected values.
        driftx = np.zeros(len(target_threeD_array[1, :, 1]))
        drifty = np.zeros(len(target_threeD_array[1, 1, :]))
        corrected_array = np.zeros((len(target_threeD_array[1, :, 1]), len(target_threeD_array[1, 1, :])))

        # Correct the for sample tilt along to the y-direction
        for j in range(len(target_threeD_array[1, :, 1])):
            drifty[j] = np.mean(array_min[j, :])
            corrected_array[j,:] = array_min[j, :] - drifty[j]

        # Correct the X drift button
        x = np.arange(xSIZE)
        if controller1 == "On":
            for i in np.arange(xSIZE):
                driftx[i] = np.mean(corrected_array[CROP: ySIZE - CROP, i])

            corrLIN = np.polyfit(x, driftx, 1)
            driftxL = x * corrLIN[0] + corrLIN[1]

            corrected_array[CROP: ySIZE - CROP, :] = corrected_array[CROP: ySIZE - CROP, :] - driftxL
        else:
            driftx = np.zeros(len(target_threeD_array[1, :, 1]))

        # Apply corrected slope to each level of 3D numpy array
        arraytotcorr = np.empty_like(target_threeD_array)
        for j in range(len(target_threeD_array[1, :, 1])):
            for i in range(len(target_threeD_array[1, 1, :])):
                arraytotcorr[:, i, j] = target_threeD_array[:, i, j] - driftx[i] - drifty[j]


        return driftx, drifty, corrected_array, arraytotcorr, ind, CROP

    def bin_array(self, arraytotcorr, ind, rawarray1, rawarray2):
        """
        Function to reduce the size of large datasets.  Data placed into equidistant bins for each x,y coordinate and
        new vector created from the mean of each bin.  Size of equidistant bins determined by 0.01 nm increments of
        Zsensor data.
        :param arraytotcorr: Zsensor data corrected for sample tilt using correct_slope function.  Important to use this
         and not raw Zsensor data so as to get an accurate Zmax value.
        :param ind: Index of Zmax for each x,y coordinate to cut data set into approach and retract.
        :param rawarray: 3D numpy array the user wishes to reduce in size (e.g. phase, amp)
        :return: 3D numpy array of binned approach values, 3D numpy array of binned retract values.
        """

        # Generate empty numpy array to populate.

        assert np.isfortran(arraytotcorr) == True, "Input Phase array not passed through generate_array fucntion.  \
                                                        Needs to be column-major indexing."

        arraymean = np.zeros(len(arraytotcorr[:, 1, 1]))

        # Create list of the mean Zsensor value for each horizontal slice of Zsensor array.
        for z in range(len(arraymean)):
            arraymean[z] = np.mean(arraytotcorr[z, :, :])

        # Turn mean Zsensor data into a linear vector with a step size of 0.02 nm.
        linearized = np.arange(-0.2, arraymean.max(), Zbin)

        # Generate empty array to populate
        reduced_array_approach1 = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        reduced_array_approach2 = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        reduced_array_retract1 = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        reduced_array_retract2 = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        # Cut raw phase/amp datasets into approach and retract, then bin data according to the linearized Zsensor data.
        # Generate new arrays from the means of each bin.  Perform on both approach and retract data.

        for i, j in itertools.product(range(len(arraytotcorr[1, 1, :])), range(len(arraytotcorr[1, :, 1]))):
            z = arraytotcorr[:(int(ind[i, j])), i, j]  # Create dataset with just retract data
            digitized = np.digitize(z, linearized)  # Bin Z data based on standardized linearized vector.
            # Populate new array with mean of binned Z data
            reduced_array_approach1[:, i, j] = \
                [np.mean(rawarray1[(np.where(digitized == n)[0]).tolist(), i, j]).tolist() for n in
                 range(len(linearized))]
            reduced_array_approach2[:, i, j] = \
                [np.mean(rawarray2[(np.where(digitized == n)[0]).tolist(), i, j]).tolist() for n in
                 range(len(linearized))]

        for i, j in itertools.product(range(len(arraytotcorr[1, 1, :])), range(len(arraytotcorr[1, :, 1]))):
            z = arraytotcorr[-(int(ind[i, j])):, i, j]  # Create dataset with just approach data.
            z = np.flipud(z)  # Flip array so surface is at the bottom on the plot.
            digitized = np.digitize(z, linearized)  # Bin Z data based on standardized linearized vector.
            # Populate new array with mean of binned Z data
            reduced_array_retract1[:, i, j] = \
                [np.mean(rawarray1[(np.where(digitized == n)[0]).tolist(), i, j]).tolist() for n in
                 range(len(linearized))]
            reduced_array_retract2[:, i, j] = \
                [np.mean(rawarray2[(np.where(digitized == n)[0]).tolist(), i, j]).tolist() for n in
                 range(len(linearized))]


        #  Merge Phase and Amp array into two different reduced array, contains approach and retract movement
        reduced_array1 = np.concatenate((reduced_array_approach1, reduced_array_retract1), axis=0)
        reduced_array2 = np.concatenate((reduced_array_approach2, reduced_array_retract2), axis=0)
        return linearized, reduced_array1, reduced_array_approach1, reduced_array_retract1, reduced_array2, \
               reduced_array_approach2, reduced_array_retract2

    def export_cleaned_data(self, file, Ddriftx, Ddrifty, Phase_reduced_array_D, Amp_reduced_array_D, valu4, Dcorrected_array, Dlinearized, export_filename0, CROP, controller2):
        """Export the cleaned HDF5 file with several exports. For more details information, please check in the Tutorial window in our GUI"""
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

        if controller2 == 'On':
            new_h5file_g1.create_dataset('Deflection', data=valu4, dtype='f4')
        else:
            pass

        new_h5file_g1.create_dataset('PHASEphaseD', data = Phase_reduced_array_D, dtype='f4')
        new_h5file_g1.create_dataset('AMPampD', data = Amp_reduced_array_D, dtype='f4')

        new_h5file_g2.create_dataset('Ddriftx', data=Ddriftx, dtype='f4')
        new_h5file_g2.create_dataset('Ddrifty', data=Ddrifty, dtype='f4')
        new_h5file_g2.create_dataset('Dlinear', data=Dlinearized, dtype='f4')
        new_h5file_g2.create_dataset('Dcorr', data=Dcorrected_array, dtype='f4')
        new_h5file_g2.create_dataset('Zbin', data=Zbin, dtype='f4')
        new_h5file_g2.create_dataset('CROP', data=CROP, dtype='f4')

        attrs_export = dict([("AmpInvOLS", AmpInvOLS), ("AmpDrive", AmpDrive), ("Qfactor", Qfactor), ("FreqDrive", FreqDrive), ("FreqRes", FreqRes), ("Xnm", Xnm), ("Ynm", Ynm)])
        dt = h5py.special_dtype(vlen=str)

        new_h5file_g3.create_dataset('METAdata', data=METAdata_convert)
        new_h5file_g3.create_dataset('Attrs_info_input_HDF5', data=attrs_export, dtype=dt)
        return new_h5file, Xnm, Ynm

    def __init__(self, parent, controller):
        """Wrapping function to link the GUI interface to the functions behind each button click."""
        global zbinsize
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')

        # Build buttons
        label1 = ttk.Label(self, text="Step 1: Data Pre-Processing", font=Huge_Font, background='#ffffff')
        label1.pack(pady=10, padx=10)

        label3 = ttk.Label(self, text='Cleaned Dataset Name', font=Large_Font, background='#ffffff')
        label3.pack(pady=10, padx=10)
        export_filename = ttk.Entry(self)
        export_filename.pack()

        label4 = ttk.Label(self, text="Correct Slope Switch", font=Large_Font, background='#ffffff')
        label4.pack(padx=5, pady=5)
        label4_1 = LabelFrame(self)
        label4_1.pack()
        listbox = Listbox(label4_1, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, 'On')
        listbox.insert(2, 'Off')
        listbox.bind('<<ListboxSelect>>', self.Curselect1)

        label5 = ttk.Label(self, text="Tip Deflection Switch", font=Large_Font, background='#ffffff')
        label5.pack(padx=5, pady=5)
        label5_1 = LabelFrame(self)
        label5_1.pack()
        listbox = Listbox(label5_1, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, 'On')
        listbox.insert(2, 'Off')
        listbox.bind('<<ListboxSelect>>', self.Curselect2)

        bin = ttk.Label(self, text="Zbin Size", background='#ffffff', font=Large_Font)
        bin.pack()
        bin_suggestion=ttk.Label(self, text="(Suggested to be 0.02)", background='#ffffff', font=Small_Font)
        bin_suggestion.pack()
        zbinsize = ttk.Entry(self)
        zbinsize.pack()

        boom = tk.Button(self, text="Load File", bg='white', command=lambda: self.openfilename())
        boom.pack(padx=5,pady=5)
        boom.config(width=15)

        button0 = tk.Button(self, text="Process Data & Export HDF5 File",bg='white', command=lambda:
        (self.get_source(export_filename, controller2), self.export_HDF5()))
        button0.pack(pady=5, padx=5)

        button2 = tk.Button(self, text="Plot Data", bg='white', command=lambda: controller.show_frame(load_data))
        button2.pack(padx=5, pady=5)
        button2.config(width=15)

        button3 = tk.Button(self, text="Tutorial", bg='white', command=lambda: controller.show_frame(tutorial))
        button3.pack(pady=5, padx=5)
        button3.config(width=15)

        button4 = tk.Button(self, text="Acknowledgements", bg='white',
                            command=lambda: controller.show_frame(acknowledge))
        button4.pack(pady=5, padx=5)

        button5 = tk.Button(self, text="Quit", bg='white', command=lambda: controller.quit())
        button5.pack(pady=5, padx=5)
        button5.config(width=15)


class load_data(tk.Frame):
    """The function for user to input the objectives for further visualizations"""
    def Curselect2(self, event):
        """The mouse click event for selecting the objectives you are interested to cleanup"""
        global valu
        widget = event.widget  # Define the event of the objective from the GUI
        select = widget.curselection()  # Read the selection from the GUI objectives
        valu = widget.get(select[0])    # Return the selection from the GUI objectives
        return valu

    def Curselect3(self, event):
        """The mouse click event for selecting the objectives you are interested to cleanup"""
        global valu0
        widget = event.widget  # Define the event of the objective from the GUI
        select = widget.curselection()  # Read the selection from the GUI objectives
        valu0 = widget.get(select[0])    # Return the selection from the GUI objectives
        return valu0

    def get_data(self, Xnm, Ynm):
        """The function to return the inputs to the GUI functions"""
        global linearized
        global x_actual
        global y_actual
        global reduced_array_retract
        global reduced_array_approach

        Xnm = (np.fromstring(Xnm, dtype=float, sep=' ') * 1e9)
        Ynm = (np.fromstring(Ynm, dtype=float, sep=' ') * 1e9)

        x_actual = int(Xnm)  # Export the actual size of x direction from the AFM measurements
        y_actual = int(Ynm)  # Export the actual size of y direction from the AFM measurements

        if valu0 =='Zsnsr':
            linearized = Zlinearized
            if valu == 'Phase':
                reduced_array_retract = Phase_reduced_array_retract
                reduced_array_approach = Phase_reduced_array_approach
            else:
                reduced_array_retract = Amp_reduced_array_retract
                reduced_array_approach = Amp_reduced_array_approach
        else:
            linearized = Dlinearized
            if valu == 'Phase':
                reduced_array_retract = Phase_reduced_array_retract_D
                reduced_array_approach = Phase_reduced_array_approach_D
            else:
                reduced_array_retract = Amp_reduced_array_retract_D
                reduced_array_approach = Amp_reduced_array_approach_D
        return linearized, x_actual, y_actual, reduced_array_retract, reduced_array_approach


    def __init__(self, parent, controller):
        """Define all the controllers in the load_data window"""
        global txtxactual
        global txtyactual

        # Building th buttons and text boxes
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 2: Input Dataset Information for Visualization", font=Huge_Font,
                          background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Select the Objectives", font=Large_Font, background='#ffffff')
        label2.pack(padx=10, pady=10)
        lab1 = LabelFrame(self)
        lab1.pack()
        listbox1 = Listbox(lab1, exportselection=0)
        listbox1.configure(height=2)
        listbox1.pack()
        listbox1.insert(1, 'Amp')
        listbox1.insert(2, 'Phase')
        listbox1.bind('<<ListboxSelect>>', self.Curselect2)

        label3 = ttk.Label(self, text="Select the Objectives", font=Large_Font, background='#ffffff')
        label3.pack(padx=10, pady=10)
        lab2 = LabelFrame(self)
        lab2.pack()
        listbox2 = Listbox(lab2, exportselection=0)
        listbox2.configure(height=2)
        listbox2.pack()
        listbox2.insert(1, 'Zsnsr')
        listbox2.insert(2, 'Drive')
        listbox2.bind('<<ListboxSelect>>', self.Curselect3)

        button0 = tk.Button(self, text="Apply Parameters", bg='white', command=lambda: self.get_data(Xnm, Ynm))
        button0.pack(pady=5, padx=5)
        button0.config(width=15)

        button1 = tk.Button(self, text="Force Curves Plot", bg='white',
                            command=lambda: controller.show_frame(Force_Curve_plot))
        button1.pack(pady=5, padx=5)
        button1.config(width=15)

        button2 = tk.Button(self, text="3D Plot", bg='white', command=lambda: controller.show_frame(threeD_plot))
        button2.pack(pady=5, padx=5)
        button2.config(width=15)

        button3 = tk.Button(self, text="2D Slicing Plot", bg='white', command=lambda: controller.show_frame(twoD_slicing))
        button3.pack(pady=5, padx=5)
        button3.config(width=15)

        button4 = tk.Button(self, text="2D Slicing Animation", bg='white',
                            command=lambda: controller.show_frame(animation_cool))
        button4.pack(pady=5, padx=5)

        button5 = tk.Button(self, text="Home", bg='white', command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=5, padx=5)
        button5.config(width=15)


class Force_Curve_plot(tk.Frame):
    """The function for user to input the objectives for further visualization interests based"""
    def Curselect4(self, event):
        """The mouse click event for selecting the objectives you are interested to cleanup"""
        global Z_directiondirection
        global Z_dirdir
        widget = event.widget
        select = widget.curselection()
        Z_directiondirection = widget.get(select[0])
        if Z_directiondirection == "Retract":
            Z_dirdir = linearized
        else:
            Z_dirdir = linearized
        return Z_dirdir, Z_directiondirection

    def num_picking_point(self, numclicks):
        """Define the number of the points you are interested in certain 3D plane"""
        global numclick
        numclick = int(numclicks.get())
        return numclick

    def location_slices(self, txtnslicesslices):
        """Define the location of the slice you are interested"""
        global location_slicesclices
        location_slicesclices = round(float(txtnslicesslices.get()), 2)  # Export the locations of slice from the GUI
        return location_slicesclices

    def pixel_converter(self, location_slicesclices):
        """Convert from the real z to pixel"""
        global location_slices_pixel
        location_slices_pixel = int(float(location_slicesclices / round(float(np.array(Z_dirdir).max()),4)) * len(Z_dirdir)) + 1
        return location_slices_pixel

    def create_pslist(self, Z_directiondirection):
        """The function that pulls out the approach or retract dataset."""
        global pslist
        if Z_directiondirection == "Retract":
            pslist = reduced_array_retract
        else:
            pslist = reduced_array_approach
        return pslist

    def plot_force(self, location_slices_pixel, numclick, x_actual, y_actual):
        """Plotting the Force Curves for the points where users are interested, and plotting with respect to the depth of the tip"""
        global aves_list
        phaseshift = (self.create_pslist(Z_directiondirection))[location_slices_pixel]
        cc = (self.create_pslist(Z_directiondirection))

        fig4, ax = plt.subplots(facecolor='white')
        #plt.imshow(phaseshift)
        x = np.linspace(init, x_actual, x_size)
        y = np.linspace(init, y_actual, y_size)
        Y, X = np.meshgrid(x, y)
        plt.scatter(X, Y, c=phaseshift, picker=5)

        clicks = []
        aves_list = [[] for i in range(numclick)]

        def onpick3(event):
            global click
            click = 0

            os.system('cls')
            click += 1
            clicks.append(click)
            index = event.ind

            for z in range(init, np.shape(cc)[0]):
                phs = cc[z]
                ps = phs.flatten()[index]
                ave = np.average(ps)
                aves_list[len(clicks) - 1].append(ave)

            aves_list_all = [*map(mean, zip(*aves_list))]

            if len(clicks) == numclick:
                fig = plt.figure(figsize=(12, 12))
                oo = np.linspace(init, Z_dirdir.max(), len(Z_dirdir))
                ax = fig.add_subplot(111)

                def my_func(oo, yy):
                    plt.plot(oo, aves_list[yy], "--")
                    plt.plot(oo, aves_list_all, color='black')
                    plt.plot(oo[location_slices_pixel], aves_list_all[location_slices_pixel], 'y*')
                    plt.axvline(x=oo[location_slices_pixel], color='r', linestyle='--')
                    plt.axhline(y=aves_list_all[location_slices_pixel], color='r', linestyle='--')
                    ax.annotate((list(zip(oo, aves_list_all))[location_slices_pixel][0], list(zip(oo, aves_list_all))[location_slices_pixel][1]),
                                (list(zip(oo, aves_list_all))[location_slices_pixel][0], list(zip(oo, aves_list_all))[location_slices_pixel][1]),
                                xytext=(-50, -100),textcoords='offset points')

                [my_func(oo, yy) for yy in np.arange(len(clicks))]

                ax.set_xlabel('The Depth of the Tip Relative to the Substrate Surface (nm)')
                ax.set_ylabel('The Phaseshifts of Each Picked Point Among the Depth of the Tip')
                plt.title("Forces Curves of the Picked Points", fontsize=15)
                # z = np.polyfit(x, aves, 1)
                # p = np.poly1d(z)
                plt.show()
                fig.savefig('Force Curves Plot')
            else:
                pass

        fig4.canvas.mpl_connect('pick_event', onpick3)
        plt.show()

    def __init__(self, parent, controller):
        """Define all the controllers in the Fore Curve window"""
        global txtnslicesslices
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 3: Force Curve Plot for Picked Points", font=Huge_Font,
                           background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Slice Location (nm)", font="Small_Font", background='#ffffff')
        label2.pack(pady=10, padx=10)
        txtnslicesslices = ttk.Entry(self)
        txtnslicesslices.pack()

        label3 = ttk.Label(self, text="Number of Points to Average", font="Small_Font", background='#ffffff')
        label3.pack(pady=10, padx=10)
        numclicks = ttk.Entry(self)
        numclicks.pack()

        label4 = ttk.Label(self, text="Select Z Direction", font='Large_Font', background='#ffffff')
        label4.pack(pady=10, padx=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Retract")
        listbox.insert(2, "Approach")
        listbox.bind('<<ListboxSelect>>', self.Curselect4)

        button0 = tk.Button(self,
                            text="Get the Force Curves Plot", bg='white', command=lambda:
            (self.location_slices(txtnslicesslices), self.num_picking_point(numclicks),
             self.pixel_converter(location_slicesclices), self.plot_force(location_slices_pixel,
                                                                          numclick, x_actual, y_actual)))
        button0.pack(padx=5, pady=5)

        button1 = tk.Button(self, text="3D Plot", bg='white', command=lambda: controller.show_frame(threeD_plot))
        button1.pack(pady=5, padx=5)
        button1.config(width=15)

        button2 = tk.Button(self, text="2D Slicing Plot", bg='white',
                            command=lambda: controller.show_frame(twoD_slicing))
        button2.pack(pady=5, padx=5)
        button2.config(width=15)

        button3 = tk.Button(self, text="2D Slicing Animation", bg='white',
                            command=lambda: controller.show_frame(animation_cool))
        button3.pack(pady=5, padx=5)

        button4 = tk.Button(self, text="Organize Dataset", bg='white', command=lambda: controller.show_frame(load_data))
        button4.pack(pady=5, padx=5)
        button4.config(width=15)

        button5 = tk.Button(self, text="Home", bg='white', command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=5, padx=5)
        button5.config(width=15)


class threeD_plot(tk.Frame):
    """The function for the 3D plot"""
    def Curselect5(self, event):
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
        CROP2 = 8
        # If the AFM cantilever moves upward and return the z axis information and the corresponding the data points
        # in that direction and also redefine the index and the sequence of the data points
        if Z_direction == "Retract":
            Z_dir = np.flip(linearized, axis=0)
            data1 = reduced_array_retract[:len(reduced_array_retract)-CROP2]
        # If the AFM cantilever moves downward and return the z axis information and the corresponding the data points
        # in that direction and also redefine the index and the sequence of the data points
        else:
            Z_dir = linearized
            data1 = reduced_array_approach[:len(reduced_array_approach)-CROP2]

        data1[np.isnan(data1)] = np.nanmin(data1)  # Replace NaN with min value of array.

        fig = plt.figure(figsize=(11, 9), facecolor='white')  # Define the figure to make a plot
        ax = fig.add_subplot(111, projection='3d')  # Define the 3d plot

        x = np.linspace(init, x_actual, len(data1[1, :, 1]))         # Define the plotting valuable x
        y = np.linspace(init, y_actual, len(data1[1, 1, :]))         # Define the plotting valuable y
        z = np.linspace(init, Z_dir.max(), len(data1[:, 1, 1]))      # Define the plotting valuable z

        X, Y = np.meshgrid(x, y)
        X1, Y1 = np.meshgrid(z, x)
        X2, Y2 = np.meshgrid(y, z)

        # Pull out just the outter most slices of the dataset.
        Z = data1[-1, :, :]
        Z1 = np.rot90(data1[:, :, 0], axes=(-2, -1))
        Z2 = data1[:, 0, :]

        # Create an empty set to populate with data
        cset = [[], [], []]

        # Populate each set with data and place the slice in the appropriate location in 3D projection graph
        cset[0] = ax.contourf(X, Y, Z, zdir='z', offset=Z_dir.max(),
                              vmin=np.min(data1), vmax=np.max(data1))

        # now, for the x-constant face, assign the contour to the x-plot-variable:
        cset[1] = ax.contourf(Z1, Y1, X1, zdir='x', offset=x_actual,
                              vmin=np.min(data1), vmax=np.max(data1))

        # likewise, for the y-constant face, assign the contour to the y-plot-variable:
        cset[2] = ax.contourf(X2, Z2, Y2, zdir='y', offset=0,
                              vmin=np.min(data1), vmax=np.max(data1))

        plt.colorbar(cset[0])                                      # Define the colorbar in the scatter plot
        ax.set_xlim(left=init, right=x_actual)                     # Define the X limit for the plot
        ax.set_ylim(top=y_actual, bottom=init)                     # Define the Y limit for the plot
        ax.set_zlim(top=np.nanmax(Z_dir), bottom=init)             # Define the Z limit for the plot
        ax.set_xlabel('X(nm)', fontsize=15)                        # Define the X label for the plot
        ax.set_ylabel('Y(nm)', fontsize=15)                        # Define the Y label for the plot
        ax.set_zlabel('Z(nm)', fontsize=15)                        # Define the Z label for the plot
        # Define the title for the plot
        ax.set_title('3D Plot for _' + str(Z_direction) + '_' + str(valu) + ' of the AFM data', fontsize=20, y=1.05)

        canvas = FigureCanvasTkAgg(fig, self)                      # Define the display figure in the window
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP)                   # Define the display region in GUI

        fig.savefig("3D Plot_" + str(Z_direction)+ str(valu) + ".png")  # Save the export figure as png file

    def __init__(self, parent, controller):
        """Define all the controllers in the 3D Plot window"""
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 4: 3D Plot", font=Huge_Font, background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label2.pack(pady=10, padx=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Retract")
        listbox.insert(2, "Approach")
        listbox.bind('<<ListboxSelect>>', self.Curselect5)

        button1 = tk.Button(self, text="Get 3D Plot", bg='white',
                            command=lambda: self.threeDplot(Z_direction, x_actual, y_actual))
        button1.pack(pady=5, padx=5)
        button1.config(width=15)

        button2 = tk.Button(self, text="Clear the Inputs", bg='white', command=lambda: self.clear())
        button2.pack(pady=5, padx=5)
        button2.config(width=15)

        button3 = tk.Button(self, text="2D Slicing Plot", bg='white',
                            command=lambda: controller.show_frame(twoD_slicing))
        button3.pack(pady=5, padx=5)
        button3.config(width=15)

        button4 = tk.Button(self, text="2D Slicing Animation", bg="white",
                            command=lambda: controller.show_frame(animation_cool))
        button4.pack(pady=5, padx=5)

        button5 = tk.Button(self, text="Organizing Dataset", bg='white',
                            command=lambda: controller.show_frame(load_data))
        button5.pack(pady=5, padx=5)
        button5.config(width=15)

        button6 = tk.Button(self, text="Home", bg='white', command=lambda: controller.show_frame(data_cleaning))
        button6.pack(pady=5, padx=5)
        button6.config(width=15)


class twoD_slicing(tk.Frame):
    """The functions for different scopes of 2D slicings"""
    global CROP3
    CROP3 = 8
    def export_filename(self, txtfilename):
        """Export the user input export filename into the GUI"""
        global export_filename2
        export_filename2 = txtfilename.get()  # Return the name of export file
        return export_filename2

    def get_bingo(self, var00):
        global bingo
        bingo = var00.get()
        return bingo

    def Curselect6(self, event):
        global Z_direction
        global Z_dir
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        if Z_direction == "Retract":
            Z_dir = linearized[: len(linearized) - CROP3]
        else:
            Z_dir = linearized[: len(linearized) - CROP3]
        return Z_dir, Z_direction

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
        """The function that pulls out the approach or retract dataset."""
        global pslist
        if Z_direction == "Retract":
            pslist = reduced_array_retract[: len(reduced_array_retract) - CROP3]
        else:
            pslist = reduced_array_approach[: len(reduced_array_approach) - CROP3]
        return pslist

    def twoDX_slicings(self, location_slices_pixel_x, export_filename2, bingo, x_actual, y_actual):
        """Plotting function for the X direction slicing"""
        global canvas1
        global canvas2

        root = tk.Toplevel(self)
        root.wm_state('zoomed')

        if location_slices_pixel_x in range(x_size + 1):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for X is between 0 to " + str(x_actual) + ".")

        As = np.array(self.create_pslist(Z_direction))[:, location_slices_pixel_x, :]  # Select the phaseshift data points in the certain slice plane
        As[np.isnan(As)] = np.nanmin(As)  # Replace NaN with min value of array.

        a = np.linspace(init, x_actual, x_size)[location_slices_pixel_x]  # Define the certain x slice in the x space
        b = np.linspace(init, y_actual, x_size)  # Define the y space
        c = Z_dir                                # Define the z space
        X, Z, Y = np.meshgrid(a, c, b)           # Create the meshgrid for the 3d space

        fig = plt.figure(figsize=(9, 9), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        # Define the fixed colorbar range based the overall phaseshift values from the input data file
        im = ax.scatter(X, Y, Z, c=As.flatten(), s=6, vmax=np.nanmax(np.array(self.create_pslist(Z_direction))),
                        vmin=np.nanmin(np.array(self.create_pslist(Z_direction))))
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

        canvas1 = FigureCanvasTkAgg(fig, master=root)  # Plot the 3D figure of 2D slicing
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_'.format(export_filename2) + str(valu) + '_Xslices.png'
        fig.savefig(setStr)

        fig1 = plt.figure(figsize=(9, 9), facecolor='white')
        plt.subplot(111)
        plt.imshow(As, aspect='auto', origin="lower", vmax=np.nanmax(np.array(self.create_pslist(Z_direction))),
                   vmin=np.nanmin(np.array(self.create_pslist(Z_direction))), extent=[init, y_actual, init, Z_dir.max()])
        plt.xlabel('Y', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D X Slicing (X=' + str(round(a, 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        #  Add and label colorbar
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig1, master=root)  # Plot the 2D figure of 2D slicing
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_'.format(export_filename2) + str(valu) + '_2d_Xslices.png'  # Define the export image name
        fig1.savefig(setStr)

        if bingo == 1:
            #Get the HDF5 file for the plotting
            h5file = export_filename2 + '_' + str(valu) + '_' + str(Z_direction) + str("_XSlicing.h5")  # Define the final name of the h5 file
            # Assuming As is a list of lists
            h = h5py.File(h5file, 'w')  # Create the empty h5 file
            h.create_dataset("data", data=As)  # Insert the data into the empty file
        else:
            pass

    def twoDY_slicings(self, location_slices_pixel_y, export_filename2, bingo, x_actual, y_actual):
        """Plotting function for the Y direction slicing"""
        global canvas1
        global canvas2
        if location_slices_pixel_y in range(y_size + 1):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for Y is between 0 to " + str(y_actual) + ".")
        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)[location_slices_pixel_y]
        c = Z_dir
        X, Z, Y = np.meshgrid(a, c, b)

        Bs = np.array(self.create_pslist(Z_direction))[:, location_slices_pixel_y, :]
        Bs[np.isnan(Bs)] = np.nanmin(Bs)  # Replace NaN with min value of array.

        root1 = tk.Toplevel(self)
        root1.state('zoomed')
        fig = plt.figure(figsize=(9, 9), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=Bs.flatten(), s=6, vmax=np.nanmax(np.array(self.create_pslist(Z_direction))),
                        vmin=np.nanmin(np.array(self.create_pslist(Z_direction))))
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

        canvas1 = FigureCanvasTkAgg(fig, master=root1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_'.format(export_filename2) + str(valu) + '_Yslices.png'
        fig.savefig(setStr)

        fig2 = plt.figure(figsize=(9, 9), facecolor='white')
        plt.subplot(111)
        plt.imshow(Bs, aspect='auto', origin="lower", vmax=np.nanmax(np.array(self.create_pslist(Z_direction))),
                   vmin=np.nanmin(np.array(self.create_pslist(Z_direction))), extent=[init, x_actual, init, Z_dir.max()])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D Y Slicing (Y=' + str(round(b, 3)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        #  Add and label colorbar
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig2, root1)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_'.format(export_filename2) + str(valu) + '_2d_Yslices.png'
        fig2.savefig(setStr)

        if bingo == 1:
            #Get the HDF5 file for the plotting
            h5file = export_filename2 + '_' + str(valu) + '_' + str(Z_direction) + str("_YSlicing.h5")
            # Assuming Bs is a list of lists
            h = h5py.File(h5file, 'w')
            h.create_dataset("data", data=Bs)
        else:
            pass

    def twoDZ_slicings(self, location_slices_pixel_z, export_filename2, bingo, x_actual, y_actual):
        """3D Plotting function for Z direction slicing"""
        global canvas1
        global canvas2
        global canvas3
        global numberinput
        if location_slices_pixel_z in range (len(Z_dir) + 2):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for Z is between 0 to " + str(np.array(Z_dir).max()) + ".")

        phaseshift = (self.create_pslist(Z_direction))[location_slices_pixel_z]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        X, Z, Y = np.meshgrid(a, Z_dir[location_slices_pixel_z], b)

        l = phaseshift
        l[np.isnan(l)] = np.nanmin(l)  # Replace NaN with min value of array.

        fig3, ax1 = plt.subplots(figsize=(9, 9), facecolor='white')
        im2 = ax1.imshow(l, vmax=np.nanmax(np.array(self.create_pslist(Z_direction))),vmin=np.nanmin(np.array(self.create_pslist(Z_direction))), extent=[init, x_actual, init, y_actual])

        mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'), formatter='i, j = {i}, {j}\nz = {z:.02g}'.format)

        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z=' + str(round(Z_dir[(location_slices_pixel_z) - 1], 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        #  Add and label colorbar
        cbar = plt.colorbar(im2)
        cbar.set_label(str(valu))

        fig = plt.figure(figsize=(9, 9), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im1 = ax.scatter(X, Y, Z, c=l.flatten(), s=6, vmax=np.nanmax(np.array(self.create_pslist(Z_direction))),
                        vmin=np.nanmin(np.array(self.create_pslist(Z_direction))))

        #  Add and label colorbar
        cbar = plt.colorbar(im1)
        cbar.set_label(str(valu))
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D Z Slicing (Z=' + str(round(Z_dir[(location_slices_pixel_z) - 1], 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)

        root2 = tk.Toplevel(self)
        root2.state('zoomed')
        canvas1 = FigureCanvasTkAgg(fig, master=root2)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT)

        h5file = export_filename2 + '_' + str(valu) + '_' + str(Z_direction) + str("_ZSlicing.h5")

        # Assuming phaseshift is a list of lists
        h = h5py.File(h5file, 'w')
        h.create_dataset("data", data=phaseshift)

        setStr = '{}_'.format(export_filename2) + str(valu) + '_Zslices.png'
        fig.savefig(setStr)

        canvas2 = FigureCanvasTkAgg(fig3, master=root2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_'.format(export_filename2) + str(valu) + '_2d_Zslices.png'
        fig3.savefig(setStr)

        if bingo == 1:
            #Get the HDF5 file for the plotting
            h5file = export_filename2 + str(Z_direction) + str("_Z.h5")
            # Assuming phaseshift is a list of lists
            h = h5py.File(h5file, 'w')
            h.create_dataset("data", data=phaseshift)
        else:
            pass

    def __init__(self, parent, controller):
        global txtnslices
        global txtzdir
        global txtfilename
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 5: 2D Slicing Plot", font='Huge_Font', background='#ffffff')
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
        listbox.insert(1, "Retract")
        listbox.insert(2, "Approach")
        listbox.bind('<<ListboxSelect>>', self.Curselect6)

        label4 = ttk.Label(self, text="Export Filename", font="Small_Font", background='#ffffff')
        label4.pack(pady=10, padx=10)
        txtfilename = ttk.Entry(self)
        txtfilename.pack()

        var00 = IntVar()
        checkbutton = Checkbutton(self, text="HDF5", variable=var00, bg='white')
        checkbutton.place(x=900, y=275)

        button1 = tk.Button(self, text="Get 2D X Slicing Plot", bg="white",
                            command=lambda: (self.get_bingo(var00),self.location_slices(txtnslices), self.export_filename(txtfilename),
                             self.pixel_converter(location_slices),
                             self.twoDX_slicings(location_slices_pixel_x, export_filename2, bingo, x_actual,y_actual)))
        button1.place(x=645, y=275)

        button2 = tk.Button(self, text="Get 2D Y Slicing Plot", bg="white",
                            command=lambda: (self.get_bingo(var00), self.location_slices(txtnslices), self.export_filename(txtfilename),
                                             self.pixel_converter(location_slices),
                                             self.twoDY_slicings(location_slices_pixel_y, export_filename2, bingo, x_actual,
                                                                y_actual)))
        button2.place(x=775, y=275)

        button3 = tk.Button(self, text="Get 2D Z Slicing Plot", bg="white", command=lambda:
        (self. get_bingo(var00), self.location_slices(txtnslices), self.export_filename(txtfilename), self.pixel_converter(location_slices),
        self.twoDZ_slicings(location_slices_pixel_z, export_filename2, bingo, x_actual, y_actual)))
        button3.place(x=645, y=310)

        button4 = tk.Button(self, text="Get Vector Slicing Plot", bg="white", command=lambda:
        (self.get_bingo(var00), self.location_slices(txtnslices), self.export_filename(txtfilename), self.pixel_converter(location_slices),
        self.plot_force(location_slices_pixel_z, x_actual, y_actual)))
        button4.place(x=775, y=310)

        button6 = tk.Button(self, text="3D Plot", bg="white", command=lambda: controller.show_frame(threeD_plot))
        button6.place(x=645, y=345)
        button6.config(width=15)

        button7 = tk.Button(self, text="2D Slicing Animation", bg="white",
                            command=lambda: controller.show_frame(animation_cool))
        button7.place(x=775, y=345)

        button8 = tk.Button(self, text="Organizing Dataset", bg="white",
                            command=lambda: controller.show_frame(load_data))
        button8.place(x=720, y=380)
        button8.config(width=15)

        button9 = tk.Button(self, text="Home", bg="white", command=lambda: controller.show_frame(data_cleaning))
        button9.place(x=720, y=415)
        button9.config(width=15)

        label5 = tk.Label(self, text="The reference level for the plots is set as zero at the substrate surface.",
                          bg="white", font=(None, 10))
        label5.place(x=595, y=450)


class animation_cool(tk.Frame):
    """
    Users input the range of Z(1nm-4nm), and choose the direction of the data (approach or retract).
    Plot and save Z, Y and X slices animation.
    """
    def Curselect7(self, event):
        """
        According to the cantilever direction users choose, make sure the z coordinate data and range.
        :return: Z_dir, Z_direction (z axis and the cantilever direction)
        """
        global Z_direction
        global Z_dir
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        if Z_direction == "Retract":
            Z_dir = linearized
        else:
            Z_dir = linearized
        return Z_dir, Z_direction

    def create_pslist(self, Z_direction):
        """The function for reshape the input data file depends on certain shape of the input data file, and also judge
         the AFM cantilever movement direction"""
        global pslist
        if Z_direction == "Retract":
            pslist = reduced_array_retract
        else:
            pslist = reduced_array_approach
        return pslist

    def get_ani_range(self, z_ani_range):  # x&y: user should choose from 0.5-2.0nm; z: user should choose from 1.0-4.0
        """
        Get the value that users input the range of Z coordinate.
        :param z_ani_range: the range of Z coordinate that they want to see
        :return: zanirange
        """
        global zanirange
        zanirange = float(z_ani_range.get())
        return zanirange

    # x_num_slice, y_num_slice, z_num_slice
    def save_Z_animation(self, Z_dir, x_actual, y_actual, x_size, y_size, zanirange):
        """
        After geting all the information, the animation will be saved as .gif file.
         :param Z_dir: the z coordinate data (nm)
        :param x_actual: the range of x coordinate (nm)
        :param y_actual: the range of y coordinate (nm)
        :param x_size: the number of x data
        :param y_size: the number of y data
        :param zanirange: the range of Z coordinate that users want to see
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=zanirange, bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('ZYX Slicing Animation for the ' + str(valu) + ' of AFM data', fontsize=18)
        # ------------------------------------------------------------------------------------------------------------
        ims = []
        for add in range(16):  # z_num_slice is the number of Z slices
            # ax.set_zlim(top=zanirange, bottom=Z_dir.min())
            a = np.linspace(init, x_actual, x_size)
            b = np.linspace(init, y_actual,y_size)
            c = Z_dir[
                (int(float(zanirange / Z_dir.max()) * len(Z_dir) // 16) * add)]  # get every page of Z_dir
            x, z, y = np.meshgrid(a, c, b)
            k = np.array(
                self.create_pslist(Z_direction))[(int(float(zanirange / Z_dir.max()) * len(Z_dir) // 16) * add), :, :]
            ims.append((ax.scatter(x, y, z, c=k.flatten(), s=6) ,))  # --------------------------------------- Z slice

        for add in range(16):
            # ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
            a = np.linspace(init, x_actual, x_size)
            b = np.linspace(init, y_actual, 64)[int(64 // 16) * add]
            c = Z_dir[: int(float(zanirange / Z_dir.max()) * len(Z_dir))]
            x, z, y = np.meshgrid(a, c, b)
            m = np.array(
                self.create_pslist(Z_direction))[init:int(float(zanirange / Z_dir.max()) * len(Z_dir)), :, int(64 // 16) * add]
            ims.append((ax.scatter(x, y, z, c=m.flatten(), s=6),))  # ---------------------------- Y slice

        for add in np.arange(16):
            # ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
            a = np.linspace(init, x_actual, x_size)[int(x_size // 16) * add]
            b = np.linspace(init, y_actual, 64)
            c = Z_dir[: int(float(zanirange / Z_dir.max()) * len(Z_dir))]
            x, z, y = np.meshgrid(a, c, b)
            n = np.array(
                self.create_pslist(Z_direction))[init:int(float(zanirange / Z_dir.max()) * len(Z_dir)),
                int(x_size // 16) * add, init:y_size]
            ims.append((ax.scatter(x, y, z, c=n.flatten(), s=6),))  # ---------------------------  X slice
        im_ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=12000, blit=True)

        im_ani.save('location.gif', writer=animation.ImageMagickFileWriter())
        return

    def clear(self):
        """
        clear all the users' input information
        :return:
        """
        z_ani_range.delete(0, END)

    def __init__(self, parent, controller):
        """
        Add and connect all the buttons and textboxs
        :param parent:
        :param controller:
        """
        global z_ani_range
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 6: 2D Slicing Animation", font=Huge_Font, background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Range of Slices: (Z,nm)", font=Large_Font, background='#ffffff')
        label2.pack(padx=5, pady=5)
        z_ani_range = ttk.Entry(self)
        z_ani_range.pack()

        label3 = ttk.Label(self, text="Select the Z Direction", font=Large_Font, background='#ffffff')
        label3.pack(pady=10, padx=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Retract")
        listbox.insert(2, "Approach")
        listbox.bind('<<ListboxSelect>>', self.Curselect7)

        button1 = tk.Button(self, text="Get input information", bg='white',
                            command=lambda: self.get_ani_range(z_ani_range))
        button1.pack(pady=10, padx=10)

        button2 = tk.Button(self, text="Save Animation", bg='white', command=lambda:
        self.save_Z_animation(Z_dir, x_actual, y_actual, x_size, y_size, zanirange))
        button2.pack(pady=10, padx=10)
        button2.config(width=15)

        button3 = tk.Button(self, text="Clear the Inputs", bg='white', command=lambda: self.clear())
        button3.pack(pady=10, padx=10)
        button3.config(width=15)

        button4 = tk.Button(self, text="2D Slicing Plot", bg='white',
                            command=lambda: controller.show_frame(twoD_slicing))
        button4.pack(pady=5, padx=5)
        button4.config(width=15)

        button5 = tk.Button(self, text="Organizing Dataset", bg='white',
                            command=lambda: controller.show_frame(load_data))
        button5.pack(pady=10, padx=10)
        button5.config(width=15)

        button6 = tk.Button(self, text="Home", bg='white', command=lambda: controller.show_frame(data_cleaning))
        button6.pack(pady=10, padx=10)
        button6.config(width=15)



class tutorial(ttk.Frame):
    """The function for making the tutorial about this GUI"""

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Tutorial", font=Huge_Font, background='#ffffff')
        label1.pack()

        label2 = ttk.Label(self, text='The Map of 3D Data Dive', font=Large_Font, background='#ffffff')
        label2.pack(padx=5, pady=5)

        label3 = ttk.Label(self, text='Step 1: Data Cleaning', font=Small_Font, background='#ffffff')
        label3.place(x=595, y=75)
        label4 = ttk.Label(self, text='\t''X Dift Switch & Export Cleaned HDF5 File',
                           font=Small_Font, background='#ffffff')
        label4.place(x=595, y=120)
        label5 = ttk.Label(self, text='Step 2: Input the Dataset Information for Visualization',
                           font=Small_Font,background='#ffffff')
        label5.place(x=595, y=165)
        label6 = ttk.Label(self, text='Step 3: Force Curves Plot for Picked Points',
                           font=Small_Font, background='#ffffff')
        label6.place(x=595, y=210)
        label7 = ttk.Label(self, text='Step 4: 3D Plotting', font=Small_Font, background='#ffffff')
        label7.place(x=595, y=255)
        label8 = ttk.Label(self, text='Step 5: 2D Slicing Plotting', font=Small_Font, background='#ffffff')
        label8.place(x=595, y=300)
        label9 = ttk.Label(self, text='\t''X Slicing, Y Slicing, Z Slicing & Vector Slicing',
                           font=Small_Font, background='#ffffff')
        label9.place(x=595, y=345)
        label10 = ttk.Label(self, text='Step 6: 3D Animation', font=Small_Font, background='#ffffff')
        label10.place(x=595, y=390)
        label11 = ttk.Label(self, text="The Demo of Our Software", font=Large_Font, background='#ffffff')
        label11.place(x=670, y=435)

        def source():
            """Export the video for this GUI"""
            os.system("D:/New/Dropbox/UW/training/Cleanroom/EPFMNMEM2016-V004900_DTH.mp4")

        mov = tk.Button(self, text="Play Video", bg='white', command=source)
        mov.place(x=730, y=475)
        mov.config(width=15)

        label12 = ttk.Label(self, text="New Export HDF5 File Layout", font=Large_Font, background='#ffffff')
        label12.place(x=660, y=515)
        photo0 = PhotoImage(file=os.path.join('D:/1UW/3ddatadiver/3ddatadiver', "new_export_HDF5.png"))
        img0 = tk.Label(self, image=photo0, background='#ffffff')
        img0.image = photo0
        img0.place(x=450 , y=540)

        button0 = tk.Button(self, text="Home", bg='white', command=lambda: controller.show_frame(data_cleaning))
        button0.place(x=730, y=750)
        button0.config(width=15)


class acknowledge(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label0 = ttk.Label(self, text="Acknowledge", font=Huge_Font, background='#ffffff')
        label0.pack(pady=10, padx=10)

        label1 = ttk.Label(self,
                            text="This software is supported by the Pacific Northwest National Lab and the"
                                 " DIRECT Program in University of Washington.",
                            background='#ffffff', font='Small_Font')
        label1.pack()

        photo1 = PhotoImage(file=os.path.join('D:/1UW/3ddatadiver/3ddatadiver', "PNNL.png"))
        photo2 = PhotoImage(file=os.path.join('D:/1UW/3ddatadiver/3ddatadiver', "UWDIRECT.png"))
        img1 = tk.Label(self, image=photo1, background='#ffffff')
        img2 = tk.Label(self, image=photo2, background='#ffffff')
        img1.image = photo1
        img2.image = photo2
        img1.pack(padx=20, pady=20)
        img2.pack(padx=20, pady=20)

        label2 = ttk.Label(self, text="The software is created by Ellen Murphy, Xueqiao Zhang, Renlong Zheng.",
                            font='Small_Font', background='#ffffff')
        label2.pack(pady=40, padx=10)

        button1 = tk.Button(self, text="Home", bg='white', command=lambda: controller.show_frame(data_cleaning))
        button1.pack()
        button1.config(width=15)


app = Sea()
app.mainloop()
