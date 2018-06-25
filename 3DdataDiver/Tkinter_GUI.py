import h5py
import itertools
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import os
import pandas as pd
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
        tk.Tk.wm_title(self, "3DdataDiver")  # Set up the name of our software
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
        """The mouse click event for selecting the objectives you are interested to cleanup"""
        global valu
        widget = event.widget  # Define the event of the objective from the GUI
        select = widget.curselection()  # Read the selection from the GUI objectives
        valu = widget.get(select[0])  # Return the selection from the GUI objectives
        return valu

    def get_source(self, source, export_filename, valu):
        """The function to return the inputs to the GUI functions"""
        global filename
        global export_filename0
        global FFM
        global Zsnsr
        global linearized
        global reduced_array_approach
        global reduced_array_retract
        filename = source.get()  # Export the input data file
        file = h5py.File(filename, "r+")
        FFM = file['FFM']
        Zsnsr = FFM['Zsnsr']
        export_filename0 = export_filename.get()
        threeD_array = self.generatearray(valu)[0]
        Zsnsr_threeD_array = self.generatearray(valu)[1]
        arraytocorr = self.correct_slope(Zsnsr_threeD_array)[0]
        indZ = self.correct_slope(Zsnsr_threeD_array)[1]
        linearized = self.bin_array(arraytotcorr, indZ, threeD_array)[0]
        reduced_array_approach = self.bin_array(arraytotcorr, indZ, threeD_array)[1]
        reduced_array_retract = self.bin_array(arraytotcorr, indZ, threeD_array)[2]
        return FFM, Zsnsr, export_filename0, threeD_array, Zsnsr_threeD_array, arraytocorr, indZ, linearized, reduced_array_approach, reduced_array_retract

    def generatearray(self, valu):
        """Function to pull single dataset from FFM object and initial formatting.  load_h5 function
        must be run prior to generate_array.

        :param target: Name of single dataset given in list keys generated from load_h5 function.
        target parameter must be entered as a string.

        Output: formatted numpy array of a single dataset.

        Example:

        Phase = generate_array('Phase')

        print(Phase[3,3,3]) = 106.05377
        """
        global Zsnsr_threeD_array
        global threeD_array
        threeD_array = np.array(FFM[valu])
        if len(threeD_array[:, 1, 1]) < len(threeD_array[1, 1, :]):
            threeD_array = threeD_array.transpose(2, 0, 1)

        Zsnsr_threeD_array = np.array(Zsnsr)
        if len(Zsnsr_threeD_array[:, 1, 1]) < len(Zsnsr_threeD_array[1, 1, :]):
            Zsnsr_threeD_array = Zsnsr_threeD_array.transpose(2, 0, 1)
        return threeD_array, Zsnsr_threeD_array

    def correct_slope(self, Zsnsr_threeD_array):
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
        # Convert matrix from meters to nanometers.
        Zsnsr_threeD_array = np.multiply(Zsnsr_threeD_array, -1000000000)
        # Replace zeros and NaN in raw data with neighboring values.  Interpolate does not work
        # as many values are on the edge of the array.
        Zsnsr_threeD_array[Zsnsr_threeD_array == 0] = np.nan
        mask = np.isnan(Zsnsr_threeD_array)
        Zsnsr_threeD_array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Zsnsr_threeD_array[~mask])
        # We have create an numpy array of the correct shape to populate.
        array_min = np.zeros((len(Zsnsr_threeD_array[1, :, 1]), len(Zsnsr_threeD_array[1, 1, :])))
        indZ = np.zeros((len(Zsnsr_threeD_array[1, :, 1]), len(Zsnsr_threeD_array[1, 1, :])))
        # Populate zero arrays with min z values at all x,y positions.  Also, populate indZ array
        # with the index of the min z values for use in correct_Zsnsr()
        for j in range(len(Zsnsr_threeD_array[1, :, 1])):
            for i in range(len(Zsnsr_threeD_array[1, 1, :])):
                array_min[j, i] = np.min(Zsnsr_threeD_array[:, i, j])
                indZ[j, i] = np.min(np.where(Zsnsr_threeD_array[:, i, j] == np.min(Zsnsr_threeD_array[:, i, j])))

        # Find the difference between the max and mean values in the z-direction for
        # each x,y point. Populate new matrix with corrected values.
        driftx = np.zeros(len(Zsnsr_threeD_array[1, :, 1]))
        drifty = np.zeros(len(Zsnsr_threeD_array[1, 1, :]))
        corrected_array = np.zeros((len(Zsnsr_threeD_array[1, :, 1]), len(Zsnsr_threeD_array[1, 1, :])))

        # Correct the for sample tilt along to the y-direction, then correct for sample tilt
        # along the x-direction.
        for j in range(len(Zsnsr_threeD_array[1, :, 1])):
            for i in range(len(Zsnsr_threeD_array[1, 1, :])):
                drifty[j] = np.mean(array_min[j, :])
                driftx[i] = np.mean(corrected_array[:, i])
                corrected_array[j, :] = array_min[j, :] - drifty[j]
                corrected_array[:, i] = corrected_array[:, i] - driftx[i]

        # Apply corrected slope to each level of 3D numpy array
        arraytotcorr = np.empty_like(Zsnsr_threeD_array)
        for j in range(len(Zsnsr_threeD_array[1, :, 1])):
            for i in range(len(Zsnsr_threeD_array[1, 1, :])):
                arraytotcorr[:, i, j] = Zsnsr_threeD_array[:, i, j] - driftx[i] - drifty[j]
        return arraytotcorr, indZ

    def bin_array(self, arraytotcorr, indZ, threeD_array):
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

        # Generate empty numpy array to populate.
        global reduced_array_approach
        global reduced_array_retract
        global linearized
        arraymean = np.zeros(len(arraytotcorr[:, 1, 1]))
        # Create list of the mean Zsensor value for each horizontal slice of Zsensor array.
        for z in range(len(arraymean)):
            arraymean[z] = np.mean(arraytotcorr[z, :, :])
        # Turn mean Zsensor data into a linear vector with a step size of 0.01 nm.
        linearized = np.flip(np.arange(0, np.max(arraymean), 0.01), axis=0)
        # Generate empty array to populate
        reduced_array_approach = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        reduced_array_retract = np.zeros((len(linearized), len(arraytotcorr[1, :, 1]), len(arraytotcorr[1, 1, :])))
        # Replace zeros and NaN in raw data with neighboring values.  Interpolate does not work
        # as many values are on the edge of the array.
        threeD_array[threeD_array == 0] = np.nan
        mask = np.isnan(threeD_array)
        threeD_array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), threeD_array[~mask])
        # Cut raw phase/amp datasets into approach and retract, then bin data according to the linearized Zsensor data.
        # Generate new arrays from the means of each bin.
        for j in range(len(arraytotcorr[1, :, 1])):
            for i in range(len(arraytotcorr[1, 1, :])):
                z = threeD_array[:(int(indZ[i, j])), i, j]
                bin_means = (np.histogram(z, len(linearized), weights=z)[0] / np.histogram(z, len(linearized))[0])
                # some bins are empty, use pandas interpolate() to fill in NaN values.
                bin_means_df = pd.DataFrame(bin_means)
                bin_means_df = bin_means_df.interpolate()
                bin_means = np.array(bin_means_df)  # Turn back into array.
                reduced_array_approach[:, i, j] = bin_means.flatten()  # Populate reduced array.

        for j in range(len(arraytotcorr[1, :, 1])):
            for i in range(len(arraytotcorr[1, 1, :])):
                z = threeD_array[-(int(indZ[i, j])):, i, j]
                bin_means = (np.histogram(z, len(linearized), weights=z)[0] / np.histogram(z, len(linearized))[0])
                # some bins are empty, use pandas interpolate() to fill in NaN values.
                bin_means_df = pd.DataFrame(bin_means)
                bin_means_df = bin_means_df.interpolate()
                bin_means = np.array(bin_means_df)  # Turn back into array.
                reduced_array_retract[:, i, j] = bin_means.flatten()  # Populate reduced array.
        return linearized, reduced_array_approach, reduced_array_retract

    def export_cleaned_data(self, reduced_array_approach, reduced_array_retract, export_filename0):
        h5file_approach = export_filename0 + str("_approach_") + str(valu) + str(
            ".h5")  # Define the final name of the h5 file

        h = h5py.File(h5file_approach, 'w')  # Create the empty h5 file
        h.create_dataset("data", data=reduced_array_approach)

        h5file_retract = export_filename0 + str("_retract_") + str(valu) + str(".h5")  # Define the final name of the h5 file

        h = h5py.File(h5file_retract, 'w')  # Create the empty h5 file
        h.create_dataset("data", data=reduced_array_retract)
        return h5file_approach, h5file_retract

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')

        label1 = ttk.Label(self, text="Step 1: Data Pre-processing", font=Huge_Font, background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Input Dataset", font=Large_Font, background='#ffffff')
        label2.pack()
        source = ttk.Entry(self)
        source.pack(pady=10, padx=10)

        label3 = ttk.Label(self, text="Select the Objectives", font=Large_Font, background='#ffffff')
        label3.pack(padx=10, pady=10)
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=5)
        listbox.pack()
        listbox.insert(1, 'Amp')
        listbox.insert(2, "Drive")
        listbox.insert(3, "Phase")
        listbox.insert(4, "Raw")
        listbox.insert(5, "Zsnsr")
        listbox.bind('<<ListboxSelect>>', self.Curselect1)

        label4 = ttk.Label(self, text='Export Clean Dataset Name', font=Large_Font, background='#ffffff')
        label4.pack()
        export_filename = ttk.Entry(self)
        export_filename.pack()

        button0 = ttk.Button(self, text="Input the Information",
                             command=lambda: (self.get_source(source, export_filename, valu)))
        button0.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Export H5 Files",
                             command=lambda: self.export_cleaned_data(reduced_array_approach, reduced_array_retract,
                                                                      export_filename0))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Organize Dataset", command=lambda: controller.show_frame(load_data))
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="Tutorial", command=lambda: controller.show_frame(tutorial))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Acknowledge", command=lambda: controller.show_frame(acknowledge))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Quit", command=lambda: controller.quit())
        button5.pack(pady=10, padx=10)


class load_data(tk.Frame):
    """The function for user to input the objectives for further visualization interests based"""

    def get_data(self, txtxsize, txtysize, txtxactual, txtyactual):
        """The function to return the inputs to the GUI functions"""
        global x_size
        global y_size
        global x_actual
        global y_actual
        x_size = int(
            txtxsize.get())  # Export the number of data points in x axis you are interested in, based on the input csv file
        y_size = int(
            txtysize.get())  # Export the number of data points in y axis you are interested in, based on the input csv file
        x_actual = int(txtxactual.get())  # Export the actual size of x direction from the AFM measurements
        y_actual = int(txtyactual.get())  # Export the actual size of y direction from the AFM measurements
        return x_size, y_size, x_actual, y_actual

    def clean(self):
        txtxsize.delete(0, END)  # Clean up the number data in x axis from the entry
        txtysize.delete(0, END)  # Clean up the number data in y aixs from the entry
        txtxactual.delete(0, END)  # Clean up the input x actual size data from the entry
        txtyactual.delete(0, END)  # Clean up the input y actual size data from the entry

    def __init__(self, parent, controller):  # Define all the controller in the load_data window
        global txtxsize
        global txtysize
        global txtxactual
        global txtyactual
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Step 2: Input Dataset Information for Visualization", font='Large_Font',
                          background='#ffffff')
        label.pack(pady=10, padx=10)

        label_2 = ttk.Label(self, text="X Actual", font="Small_Font", background='#ffffff')
        label_2.pack()
        txtxactual = ttk.Entry(self)
        txtxactual.pack()

        label_3 = ttk.Label(self, text="Y Actual", font="Small_Font", background='#ffffff')
        label_3.pack()
        txtyactual = ttk.Entry(self)
        txtyactual.pack()

        label_4 = ttk.Label(self, text="X Size", font="Small_Font", background='#ffffff')
        label_4.pack()
        txtxsize = ttk.Entry(self)
        txtxsize.pack()

        label_5 = ttk.Label(self, text="Y Size", font="Small_Font", background='#ffffff')
        label_5.pack()
        txtysize = ttk.Entry(self)
        txtysize.pack()

        button0 = ttk.Button(self, text="Get Dataset Information",command=lambda: self.get_data(txtxsize, txtysize, txtxactual, txtyactual))
        button0.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="3D Plot", command=lambda: controller.show_frame(threeD_plot))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="2D Slicing", command=lambda: controller.show_frame(twoD_slicing))
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="2D Slicing Animation", command=lambda: controller.show_frame(animation))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: (self.clean(), self.update_idletasks()))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)


class threeD_plot(tk.Frame):
    """The function for the 3D plot"""

    def Curselect2(self, event):
        """Export the Z_direction information for the AFM cantilever motion from the GUI"""
        global Z_direction
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        return Z_direction

    def clear(self):
        canvas.get_tk_widget().destroy()  # Clean up the export the figure in window

    def threeDplot(self, Z_direction, x_actual, y_actual, x_size, y_size):
        """3D plot function"""
        global canvas
        if Z_direction == "Up":                         # If the AFM cantilever moves upward and return the z axis information and the corresponding the data points in that direction and also redefine the index and the sequence of the data points
            Z_dir = np.flip(linearized, axis=0)
            data1 = reduced_array_retract
        else:                                           # If the AFM cantilever moves downward and return the z axis information and the corresponding the data points in that direction and also redefine the index and the sequence of the data points
            Z_dir = linearized
            data1 = reduced_array_approach

        x = np.linspace(init, x_actual, x_size)         # Define the plotting valuable x
        y = np.linspace(init, y_actual, y_size)         # Define the plotting valuable y
        z = np.linspace(init, Z_dir.max(), len(Z_dir))  # Define the plotting valuable z

        xi,zi,yi = np.meshgrid(x,z,y)

        fig = plt.figure(figsize=(11, 9), facecolor='white')               #Define the figure to make a plot
        ax = fig.add_subplot(111, projection='3d')      #Define the 3d plot
        # Define the scatter plot
        im = ax.scatter(xi, yi, zi, c=data1.flatten(), alpha=0.1, vmax=np.nanmax(data1), vmin=np.nanmin(data1))
        plt.colorbar(im)                                # Define the colorbar in the scatter plot
        ax.set_xlim(left=init, right=x_actual)          # Define the X limit for the plot
        ax.set_ylim(top=y_actual, bottom=init)          # Define the Y limit for the plot
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())# Define the Z limit for the plot
        ax.set_xlabel('X(nm)', fontsize=15)             # Define the X label for the plot
        ax.set_ylabel('Y(nm)', fontsize=15)             # Define the Y label for the plot
        ax.set_zlabel('Z(nm)', fontsize=15)             # Define the Z label for the plot
        # Define the title for the plot
        ax.set_title('3D Plot for _' + str(Z_direction) + '_' + str(valu) + ' of the AFM data', fontsize=20, y=1.05)

        canvas = FigureCanvasTkAgg(fig, self)           # Define the display figure in the window
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP)        # Define the display region in GUI

        fig.savefig("3D Plot_" + str(Z_direction)+ str(valu) + ".tif")  # Save the export figure as tif file

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Step 3: 3D Plot", font='Large_Font', background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label2.pack()
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Up")
        listbox.insert(2, "Down")
        listbox.bind('<<ListboxSelect>>', self.Curselect2)

        button1 = ttk.Button(self, text="Get 3D Plot", command=lambda: self.threeDplot(Z_direction, x_actual, y_actual, x_size, y_size))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Clear the Inputs", command=lambda: self.clear())
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="2D Slicing", command=lambda: controller.show_frame(twoD_slicing))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)


class twoD_slicing(tk.Frame):
    """The functions for different scopes of 2D slicings"""

    def export_filename(self, txtfilename):
        """Export the user input export filename into the GUI"""
        global export_filename2
        export_filename2 = txtfilename.get()  # Return the name of export file
        return export_filename2

    def Curselect3(self, event):
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
        location_slices_pixel_z = int(float(location_slices / round(float(np.array(Z_dir).max()), 2)) * len(Z_dir)) + 1
        return location_slices_pixel_x, location_slices_pixel_y, location_slices_pixel_z

    def clear(self):
        """Clear up the inputs"""
        txtnslices.delete(0, END)
        txtfilename.delete(0, END)
        canvas1.get_tk_widget().destroy()
        canvas2.get_tk_widget().destroy()

    def create_pslist(self, Z_direction):
        """The function for reshape the input data file depends on certain shape of the input data file, and also judge the AFM cantilever movement direction"""
        global pslist
        if Z_direction == "Up":
            pslist = reduced_array_retract
        else:
            pslist = np.flip(reduced_array_approach, axis=0)
        return pslist

    def twoDX_slicings(self, location_slices_pixel_x, export_filename2, x_actual, y_actual, x_size, y_size):
        """Plotting function for the X direction slicing"""
        global canvas1
        global canvas2
        if location_slices_pixel_x in range(x_size + 1):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for X is between 0 to " + str(x_actual) + ".")
        a = np.linspace(init, x_actual, x_size)[location_slices_pixel_x]  # Define the certain x slice in the x space
        b = np.linspace(init, y_actual, y_size)  # Define the y space
        c = Z_dir  # Define the z space
        X, Z, Y = np.meshgrid(a, c, b)  # Create the meshgrid for the 3d space

        As = np.array(self.create_pslist(Z_direction))[:, location_slices_pixel_x, :]  # Select the phaseshift data points in the certain slice plane

        fig = plt.figure(figsize=(9, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=As, s=6, alpha=0.2, vmax=np.array(self.create_pslist(Z_direction)).max(), vmin=np.array(self.create_pslist(Z_direction)).min())  # Define the fixed colorbar range based the overall phaseshift values from the input data file
        cbar = plt.colorbar(im)
        cbar.set_label(str(valu))  # Label the colorbar
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
        plt.imshow(As, aspect='auto', origin="lower", vmax=np.array(self.create_pslist(Z_direction)).max(),vmin=np.array(self.create_pslist(Z_direction)).min())
        plt.axis([init, y_size - 1, init, len(Z_dir) - 1])  # Adjust the axis range for the 2D slicing
        plt.xlabel('Y', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D X Slicing (X=' + str(round(a, 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig1, self)  # Plot the 2D figure of 2D slicing
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Xslices.tif'.format(export_filename2)  # Define the export image name
        fig1.savefig(setStr)

    def twoDY_slicings(self, location_slices_pixel_y, export_filename2, x_actual, y_actual, x_size, y_size):
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

        Bs = np.array(self.create_pslist(Z_direction))[init:len(Z_dir), :, location_slices_pixel_y]

        fig = plt.figure(figsize=(9, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=Bs, s=6, alpha=0.2, vmax=np.array(self.create_pslist(Z_direction)).max(), vmin=np.array(self.create_pslist(Z_direction)).min())
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
        plt.axis([init, x_size - 1, init, len(Z_dir) - 1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D Y Slicing (Y=' + str(round(b, 3)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig2, self)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Yslices.tif'.format(export_filename2)
        fig2.savefig(setStr)

    def twoDZ_slicings(self, location_slices_pixel_z, export_filename2, x_actual, y_actual, x_size, y_size):
        """3D Plotting function for Z direction slicing"""
        global canvas1
        if location_slices_pixel_z in range (len(Z_dir)+1):
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range for Z is between 0 to " + str(np.array(Z_dir).max()) + ".")

        phaseshift = (self.create_pslist(Z_direction))[location_slices_pixel_z - 1]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        X, Z, Y = np.meshgrid(a, Z_dir[(location_slices_pixel_z) - 1], b)
        l = phaseshift

        fig = plt.figure(figsize=(9, 11), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=l, s=6, vmax=np.array(self.create_pslist(Z_direction)).max(), vmin=np.array(self.create_pslist(Z_direction)).min())
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

    def twoZ_slicings(self, location_slices_pixel_z, export_filename2, x_actual, y_actual, x_size, y_size):
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
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig, self)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig1 = plt.figure(figsize=(9, 9), facecolor='white')
        plt.imshow(l, vmax=np.array(self.create_pslist(Z_direction)).max(),  vmin=np.array(self.create_pslist(Z_direction)).min())
        plt.axis([init, x_size - 1, init, y_size - 1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z=' + str(round(Z_dir[(location_slices_pixel_z) - 1], 4)) + 'nm) for the ' + str(valu) + ' of AFM data', fontsize=13)
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
        label1 = ttk.Label(self, text="Step 4: 2D Slicing", font='Large_Font', background='#ffffff')
        label1.pack()

        label_1 = ttk.Label(self, text="Slices Location", font="Small_Font", background='#ffffff')
        label_1.pack()
        txtnslices = ttk.Entry(self)
        txtnslices.pack()

        label_2 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label_2.pack()
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Up")
        listbox.insert(2, "Down")
        listbox.bind('<<ListboxSelect>>', self.Curselect3)

        label_3 = ttk.Label(self, text="Export Filename", font="Small_Font", background='#ffffff')
        label_3.pack()
        txtfilename = ttk.Entry(self)
        txtfilename.pack()

        button0 = ttk.Button(self, text="Get Location Slices & Directions",
                             command=lambda: (self.location_slices(txtnslices), self.export_filename(txtfilename), self.pixel_converter(location_slices)))
        button0.pack()

        button1 = ttk.Button(self, text="Get 2D X Slicing Plot",
                             command=lambda: self.twoDX_slicings(location_slices_pixel_x, export_filename2, x_actual, y_actual,
                                                                 x_size, y_size))
        button1.pack()

        button2 = ttk.Button(self, text="Get 2D Y Slicing Plot",
                             command=lambda: self.twoDY_slicings(location_slices_pixel_y, export_filename2, x_actual, y_actual,
                                                                 x_size, y_size))
        button2.pack()

        button3 = ttk.Button(self, text="Get 2D Z Slicing Plot", command=lambda: (self.twoDZ_slicings(location_slices_pixel_z, export_filename2, x_actual, y_actual, x_size, y_size), self.twoZ_slicings(location_slices_pixel_z, export_filename2, x_actual, y_actual, x_size, y_size)))
        button3.pack()

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: (self.clear()))
        button4.pack()

        button5 = ttk.Button(self, text="3D Plot", command=lambda: controller.show_frame(threeD_plot))
        button5.pack()

        button6 = ttk.Button(self, text="2D Slicing Animation", command=lambda: controller.show_frame(animation))
        button6.pack()

        button7 = ttk.Button(self, text="Organize Dataset", command=lambda: controller.show_frame(load_data))
        button7.pack()

        button8 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button8.pack()

        label2 = ttk.Label(self, text="The reference level for the plots is set as zero at the substrate surface.",
                           font=(None, 10))
        label2.pack()


def Curselect4(event1):
    global Z_direction
    widget = event1.widget
    select = widget.curselection()
    Z_direction = widget.get(select[0])


def Judge_Z_direction(Z_direction):
    global Z_dir
    if Z_direction == "Up":
        Z_dir = np.flip(linearized, axis=0)
    else:
        Z_dir = linearized


class animation(tk.Frame):
    global canvas4
    global event1
    global update_animation

    def slice(self, numslices):
        slice = int(numslices.get())
        return slice

    def Curselect5(self, event):  # Slicing Directions
        global Dir
        widget = event.widget
        select = widget.curselection()
        Dir = widget.get(select[0])
        return Dir

    def callback(self):
        global update_animation
        update_animation = True

    def clear(self):
        numslices.delete(0, END)
        txtfilename2.delete(0, END)
        canvas4.get_tk_widget().destroy()

    def __init__(self, parent, controller):
        global numslices
        global txtfilename2
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Step 5: 2D Slicing Animation", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)
        label0 = ttk.Label(self, text="Animation is under construction. It will come soon!", font="Large_Font",
                           background='#ffffff')
        label0.pack(padx=10, pady=10)

        label1 = ttk.Label(self, text="Number of Slices", font='Large_Font', background='#ffffff')
        label1.pack()
        numslices = ttk.Entry(self)
        numslices.pack()

        label2 = ttk.Label(self, text="Select the Z Direction", font='Large_Font', background='#ffffff')
        label2.pack()
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=2)
        listbox.pack()
        listbox.insert(1, "Up")
        listbox.insert(2, "Down")
        listbox.bind('<<ListboxSelect>>', Curselect4)

        label3 = ttk.Label(self, text="Select Animation Direction", font='Large_Font', background='#ffffff')
        label3.pack()
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=3)
        listbox.pack()
        listbox.insert(1, "X")
        listbox.insert(2, "Y")
        listbox.insert(3, "Z")
        listbox.bind('<<ListboxSelect>>', self.Curselect5)

        label4 = ttk.Label(self, text="Export Filename", font="Small_Font", background='#ffffff')
        label4.pack()
        txtfilename2 = ttk.Entry(self)
        txtfilename2.pack()

        button1 = ttk.Button(self, text="Get Inputs",
                             command=lambda: (self.numslice(numslices), Judge_Z_direction(Z_direction)))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Get Z Animation")
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="Clear the Inputs", command=lambda: self.clear())
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)


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
                            text="This software is supported by the Pacific Northwest National Lab and the DIRECT Program in University of Washington.",
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
