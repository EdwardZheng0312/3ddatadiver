import h5py
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import os
import pandas as pd
import sys
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


Large_Font = ("Vardana", 15)
Small_Font = ("Vardana", 11)
Tiny_Font = ("Vardana", 8)
init = 0


class Sea(tk.Tk):
    """Control function for the windows translate """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.tk.call('wm', 'iconphoto', self._w, PhotoImage(file='taiji.png'))   #Set up the iconphoto for our software
        tk.Tk.wm_title(self, "High-Resolution AFM 3D Visualization Client")      #Set up the name of our software
        self.state('zoomed')                                                     #Set up the inital window operation size

        container = tk.Frame(self)                                               #Define the properties of the bracket of windows inside the GUI
        container.pack(side="top", fill="both", expand=True)                     #Define the layout of the bracket
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}                                                         #Define the windows in the GUI
        for F in (data_cleaning, load_data, threeD_plot, twoD_slicing, animation, tutorial, acknowledge):
            frame = F (container, self)                                          #Call the certain window from the bracket

            self.frames[F] = frame                                               #Define each son windows in GUI
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(data_cleaning)                                           #Call the main window of the GUI

    def show_frame(self,cont):                                                   #Show the call window
        frame = self.frames[cont]
        frame.tkraise()                                                          #Raise the called window to the top


class data_cleaning(tk.Frame):
    """Data Preprocessing function to clean up the input data file and export the cleaned dataset"""
    def Curselect1(self, event):
        """The mouse click event for selecting the objectives you are interested to cleanup"""
        global valu
        widget = event.widget                                                    #Define the event of the objective from the GUI
        select = widget.curselection()                                           #Read the selection from the GUI objectives
        valu = widget.get(select[0])                                             #Return the selection from the GUI objectives
        return valu

    def get_source(self, source):
        """The function to return the inputs to the GUI functions"""
        global filename
        global data
        global z
        global z_approach
        global z_retract
        filename = source.get()                                                   #Export the input data file
        data = self.load_data(filename)[0]                                        #Export the input data from the data file
        z = self.load_data(filename)[1]                                           #Export the z axis data
        z_approach = self.load_data(filename)[2]                                  #Export the Z axis data for AFM cantilever goes downward the substrate surface
        z_retract = self.load_data(filename)[3]                                   #Export the z axis data for AFM cantilever goes upward the substrate surface
        return filename, data, z, z_approach, z_retract

    def load_data(self, filename):
        """The function to import and primary data pre cleaning process for the input data file."""
        data = pd.read_csv(filename)                                              #Read the csv file
        z = data.iloc[:, 0]                                                       #Define the data range in the input csv file

        z_retractt = z[: len(z)//2][::-1]                                         #Define the Z axis data for AFM cantilever goes upward the substrate surface from the csv file
        z_approach = z[len(z)//2:][::-1]                                          #Define the Z axis data for AFM cantilever goes downward the substrate surface from the csv file

        z_approach = z_approach.reset_index(drop=True)                            #Define the Z axis approach data index starts from 1
        z_retract = z_retractt.reset_index(drop=True)                             #Define the Z axis retract datat index starts from 1
        return data, z, z_approach, z_retract

    def matconvertor(self):
        #mat = scipy.io.loadmat()
        #phase = mat['PHASEtot']
        return

    def Zsnsr_correction(self):

        return

    def clear(self):

        return

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')

        label1 = ttk.Label(self, text="Step 1: Data Pre-processing", font='Large_Font', background='#ffffff')
        label1.pack(pady=10, padx=10)

        label2 = ttk.Label(self, text="Input Dataset",font="Small_Font", background='#ffffff')
        label2.pack()
        source =ttk.Entry(self)
        source.pack(pady=10, padx=10)

        button0 = ttk.Button(self, text="Get Dataset", command=lambda: (self.get_source(source)))
        button0.pack(pady=10, padx=10)

        label3 = ttk.Label(self, text="Select the Objectives", font='Large_Font', background='#ffffff')
        label3.pack()
        lab = LabelFrame(self)
        lab.pack()
        listbox = Listbox(lab, exportselection=0)
        listbox.configure(height=4)
        listbox.pack()
        listbox.insert(1, "Phase Shift")
        listbox.insert(2, "Tip Deflection")
        listbox.insert(3, "Amplitude")
        listbox.insert(4, "Frequency")
        listbox.bind('<<ListboxSelect>>', self.Curselect1)

        button1 = ttk.Button(self, text="Data Preprocessing",command=lambda: controller.self)
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
        x_size = int(txtxsize.get())                                              #Export the number of data points in x axis you are interested in, based on the input csv file
        y_size = int(txtysize.get())                                              #Export the number of data points in y axis you are interested in, based on the input csv file
        x_actual = int(txtxactual.get())                                          #Export the actual size of x direction from the AFM measurements
        y_actual = int(txtyactual.get())                                          #Export the actual size of y direction from the AFM measurements
        return x_size, y_size, x_actual, y_actual

    def clean(self):
        txtxsize.delete(0,END)                                                    #Clean up the number data in x axis from the entry
        txtysize.delete(0,END)                                                    #Clean up the number data in y aixs from the entry
        txtxactual.delete(0,END)                                                  #Clean up the input x actual size data from the entry
        txtyactual.delete(0,END)                                                  #Clean up the input y actual size data from the entry

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):                                                   #Set up the range of Z axis you are interested in
            phaseshift = data.iloc[k, 1:]                                         #Select the data points from zero row to the end row, from second column to the last column]
            ps = np.array(phaseshift)                                             #Convert from the list to the numpy array
            ps_reshape = np.reshape(ps, (x_size, y_size))                         #Reshape the data into the user defined array
            pslist.append(ps_reshape)                                             #Export all the reshaped data points for each z value into one list
        return pslist

    def __init__(self, parent, controller):                                       #Define all the controller in the load_data window
        global txtxsize
        global txtysize
        global txtxactual
        global txtyactual
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Step 2: Input Dataset Information for Visualization", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label_2 = ttk.Label(self, text="X Actual",font="Small_Font", background='#ffffff')
        label_2.pack()
        txtxactual =ttk.Entry(self)
        txtxactual.pack()

        label_3 = ttk.Label(self, text="Y Actual",font="Small_Font", background='#ffffff')
        label_3.pack()
        txtyactual =ttk.Entry(self)
        txtyactual.pack()

        label_4 = ttk.Label(self, text="X Size",font="Small_Font", background='#ffffff')
        label_4.pack()
        txtxsize =ttk.Entry(self)
        txtxsize.pack()

        label_5 = ttk.Label(self, text="Y Size",font="Small_Font", background='#ffffff')
        label_5.pack()
        txtysize =ttk.Entry(self)
        txtysize.pack()

        button0 = ttk.Button(self, text="Get Dataset Information", command=lambda: self.get_data(txtxsize, txtysize, txtxactual, txtyactual))
        button0.pack(pady=10, padx=10)

        button1 =ttk.Button(self, text="3D Plot", command=lambda: controller.show_frame(threeD_plot))
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

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):
            phaseshift = data.iloc[k, 1:]
            ps = np.array(phaseshift)
            ps_reshape = np.reshape(ps, (x_size, y_size))
            pslist.append(ps_reshape)
        return pslist

    def threeDplot(self, Z_direction, z, x_actual, y_actual, x_size, y_size):
        """3D plot function"""
        global canvas
        if Z_direction == "Up":                                                     #If the AFM cantilever moves upward and return the z axis information and the corresponding the data points in that direction and also redefine the index and the sequence of the data points
            Z_dir = data.iloc[:,0].iloc[:len(z) // 2][::-1].reset_index(drop=True)
            data1 = data.iloc[:,:].iloc[:len(z) // 2].drop(['Z (nm)'], axis=1)[::-1].reset_index(drop=True)
        else:                                                                       #If the AFM cantilever moves downward and return the z axis information and the corresponding the data points in that direction and also redefine the index and the sequence of the data points
            Z_dir = data.iloc[:,0].iloc[len(z) // 2:].reset_index(drop=True)
            data1=data.iloc[:,:].iloc[len(z) // 2:].drop(['Z (nm)'], axis=1).reset_index(drop=True)

        retract_as_numpy = data1.as_matrix(columns=None)                            #Convert the data to the maxtrix
        retract_as_numpy_reshape1 = retract_as_numpy.reshape(len(Z_dir), x_size, y_size)    #Reshape the data
        retract_as_numpy_reshape2 = retract_as_numpy_reshape1.flatten('F')          #Convert the data points into one list with the same sequencec

        x = np.linspace(init, x_actual, x_size)                                     #Define the plotting valuable x
        y = np.linspace(init, y_actual, y_size)                                     #Define the plotting valuable y
        z = np.linspace(init, Z_dir.max(), len(Z_dir))                              #Define the plotting valuable z

        points = []
        for element in itertools.product(x, y, z):                                  #Creates a "flat" list representation of a 3D space
            points.append(element)

        fxyz = list(retract_as_numpy_reshape2)                                      #Convert the flatten data points list into another list
        xi, yi, zi = zip(*points)                                                   #Zipped the flatten data points into the represented 3D space one by one in the same order as the flatten data points list

        fig = plt.figure(figsize=(11, 9))                                           #Define the figure to make a plot
        ax = fig.add_subplot(111, projection='3d')                                  #Define the 3d plot
        im = ax.scatter(xi, yi, zi, c=fxyz, alpha=0.2)                              #Define the scatter plot
        plt.colorbar(im)                                                            #Define the colorbar in the scatter plot
        ax.set_xlim(left=init,right=x_actual)                                       #Define the X limit for the plot
        ax.set_ylim(top=y_actual, bottom=init)                                      #Define the Y limit for the plot
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())                            #Define the Z limit for the plot
        ax.set_xlabel('X(nm)', fontsize=15)                                         #Define the X label for the plot
        ax.set_ylabel('Y(nm)', fontsize=15)                                         #Define the Y label for the plot
        ax.set_zlabel('Z(nm)', fontsize=15)                                         #Define the Z label for the plot
        ax.set_title('3D Plot for ' +str(valu)+ ' of the AFM data', fontsize=20)    #Define the title for the plot

        canvas = FigureCanvasTkAgg(fig, self)                                       #Define the display figure in the window
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP)                                    #Define the display region in GUI

        fig.savefig("3D Plot.tif")                                                  #Save the export figure as tif file

    def clear(self):
        txtzdir.delete(0,END)                                                       #Clean up the input for the Z direction
        canvas.get_tk_widget().destroy()                                            #Clean up the export the figure in window

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

        button1 = ttk.Button(self, text="Get 3D Plot", command=lambda: self.threeDplot(Z_direction, z, x_actual, y_actual, x_size, y_size))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Clear the Inputs", command=lambda: self.clear())
        button2.pack(pady=10, padx=10)

        button3 =ttk.Button(self, text="2D Slicing", command=lambda:controller.show_frame(twoD_slicing))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)


class twoD_slicing(tk.Frame):
    """The functions for different scopes of 2D slicings"""
    def location_slices(self, txtnslices):
        """Define the location of the slice you are interested"""
        global location_slices
        location_slices = int(txtnslices.get())                                       #Export the locations of slice from the GUI
        if location_slices in range(1, len(Z_dir)+1):                                 #Testing for the input whether or not beyond the input data file
            pass
        else:
            tkMessageBox.askretrycancel("Input Error", "Out of range, The expected range is between 1 to "+str(len(Z_dir))+".")               #If the input the beyond the range, pop-up the error message
        return location_slices

    def export_filename(self, txtfilename):
        """Export the user input export filename into the GUI"""
        global export_filename2
        export_filename2 = txtfilename.get()                                          #Return the name of export file
        return export_filename2

    def Curselect3(self, event):
        global Z_direction
        global Z_dir
        widget = event.widget
        select = widget.curselection()
        Z_direction = widget.get(select[0])
        if Z_direction == "Up":
            Z_dir = z_retract
        else:
            Z_dir = z_approach
        return Z_dir

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file, and also judge the AFM cantilever movement direction"""
        pslist = []
        if Z_direction == "Up":
            for k in range(len(z)//2-1, -1, -1):
                phaseshift = data.iloc[k, 1:]
                ps = np.array(phaseshift)
                ps_reshape = np.reshape(ps, (x_size, y_size))
                pslist.append(ps_reshape)
        else:
            for k in range(len(z)-1, len(z)//2-1, -1):
                phaseshift = data.iloc[k, 1:]
                ps = np.array(phaseshift)
                ps_reshape = np.reshape(ps, (x_size, y_size))
                pslist.append(ps_reshape)
        return pslist

    def twoDX_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        """Plotting function for the X direction slicing"""
        global canvas1
        global canvas2
        a = np.linspace(init, x_actual, x_size)[location_slices]                     #Define the certain x slice in the x space
        b = np.linspace(init, y_actual, y_size)                                      #Define the y space
        c = Z_dir                                                                    #Define the z space
        X, Z, Y = np.meshgrid(a, c, b)                                               #Create the meshgrid for the 3d space

        As = np.array(self.create_pslist(x_size, y_size))[:, location_slices, :]     #Select the phaseshift data points in the certain slice plane

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=As, s=6, alpha=0.2, vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())   #Define the fixed colorbar range based the overall phaseshift values from the input data file
        cbar = plt.colorbar(im)
        cbar.set_label(str(valu))                                                    #Label the colorbar
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D X Slicing (X='+str(round(a,4)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)

        canvas1 = FigureCanvasTkAgg(fig, self)                                       #Plot the 3D figure of 2D slicing
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h5file = export_filename2+str(".h5")                                         #Define the final name of the h5 file

        # Assuming As is a list of lists
        h = h5py.File(h5file, 'w')                                                   #Create the empty h5 file
        h.create_dataset("data", data=As)                                            #Insert the data into the empty file

        fig1 = plt.figure(figsize=(11, 9))
        plt.subplot(111)
        plt.imshow(As, aspect='auto', origin="lower", vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())
        plt.axis([init, y_size-1, init, len(Z_dir)-1])                               #Adjust the axis range for the 2D slicing
        plt.xlabel('Y', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D X Slicing (X='+str(round(a,4)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig1, self)                                     #Plot the 2D figure of 2D slicing
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Xslices.tif'.format(export_filename2)                       #Define the export image name
        fig1.savefig(setStr)

    def twoDY_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        """Plotting function for the Y direction slicing"""
        global canvas1
        global canvas2
        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)[location_slices]
        c = Z_dir
        X, Z, Y = np.meshgrid(a, c, b)

        Bs = np.array(self.create_pslist(x_size, y_size))[init:len(Z_dir), :, location_slices]

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=Bs, s=6, alpha=0.2, vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())
        cbar = plt.colorbar(im)
        cbar.set_label(str(valu))
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D Y Slicing (Y='+str(round(b,3)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)

        canvas1 = FigureCanvasTkAgg(fig, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h5file = export_filename2+str(".h5")

        # Assuming Bs is a list of lists
        h = h5py.File(h5file, 'w')
        h.create_dataset("data", data=Bs)

        setStr = '{}_Yslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig2 = plt.figure(figsize=(11, 9))
        plt.subplot(111)
        plt.imshow(Bs, aspect='auto', vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())
        plt.axis([init, x_size-1, init, len(Z_dir)-1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D Y Slicing (Y='+str(round(b,3)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig2, self)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Yslices.tif'.format(export_filename2)
        fig2.savefig(setStr)

    def twoDZ_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        """3D Plotting function for Z direction slicing"""
        global canvas1
        phaseshift = (self.create_pslist(x_size, y_size))[location_slices-1]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        X, Z, Y = np.meshgrid(a, Z_dir[(location_slices)-1], b)
        l = phaseshift

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=l, s=6, vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())
        cbar = plt.colorbar(im)
        cbar.set_label(str(valu))
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('3D Z Slicing (Z='+str(round(Z_dir[(location_slices)-1],4)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)

        canvas1 = FigureCanvasTkAgg(fig, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h5file = export_filename2+str(".h5")

        # Assuming phaseshift is a list of lists
        h = h5py.File(h5file, 'w')
        h.create_dataset("data", data=phaseshift)


        setStr = '{}_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)


    def twoZ_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        """2D Plotting function for Z direction slicing"""
        global canvas2
        phaseshift = (self.create_pslist(x_size, y_size))[location_slices-1]

        l = phaseshift

        fig = plt.figure(figsize=(9, 9))
        plt.imshow(l, vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())
        plt.axis([init, x_size-1, init, y_size-1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z='+str(round(Z_dir[(location_slices)-1],4)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)
        cbar = plt.colorbar()
        cbar.set_label(str(valu))

        canvas2 = FigureCanvasTkAgg(fig, self)
        canvas2.show()
        canvas2.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig1 = plt.figure(figsize=(9, 9))
        plt.imshow(l, vmax=np.array(self.create_pslist(x_size, y_size)).max(), vmin=np.array(self.create_pslist(x_size, y_size)).min())
        plt.axis([init, x_size-1, init, y_size-1])
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z='+str(round(Z_dir[(location_slices)-1],4)) + 'nm) for the '+str(valu)+' of AFM data', fontsize=13)
        plt.colorbar()
        x = np.array(plt.ginput(1))
        s=int((x[0][0]/24)//(x_actual/(x_size-1)))
        d=int((x[0][1]/24)//(y_actual/(y_size-1)))
        print(l[s][d])
        canvas3 = FigureCanvasTkAgg(fig1, self)
        canvas3.show()

    def clear(self):
        """Clear up the inputs"""
        txtnslices.delete(0, END)
        txtfilename.delete(0,END)
        canvas1.get_tk_widget().destroy()
        canvas2.get_tk_widget().destroy()

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

        button0 = ttk.Button(self, text="Get Location Slices & Directions", command=lambda: (self.location_slices(txtnslices), self.export_filename(txtfilename)))
        button0.pack()

        button1 = ttk.Button(self, text="Get 2D X Slicing Plot", command=lambda: self.twoDX_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size))
        button1.pack()

        button2 = ttk.Button(self, text="Get 2D Y Slicing Plot", command=lambda: self.twoDY_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size))
        button2.pack()

        button3 = ttk.Button(self, text="Get 2D Z Slicing Plot", command=lambda: (self.twoDZ_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size), self.twoZ_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size)))
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

        label2 = ttk.Label(self, text="The reference level for the plots is set as zero at the substrate surface.", font=(None,10))
        label2.pack()


def Curselect4(event1):
    global Z_direction
    widget = event1.widget
    select = widget.curselection()
    Z_direction = widget.get(select[0])

def Judge_Z_direction(Z_direction):
    global Z_dir
    if Z_direction == "Up":
        Z_dir = z_retract
    else:
        Z_dir = z_approach

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
        cbar4.remove()
        canvas4.get_tk_widget().destroy()

    def __init__(self, parent, controller):
        global numslices
        global txtfilename2
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Step 5: 2D Slicing Animation", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label1 =ttk.Label(self, text="Number of Slices", font='Large_Font', background='#ffffff')
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

        button1 = ttk.Button(self, text="Get Inputs", command=lambda: (self.numslice(numslices), Judge_Z_direction(Z_direction)))
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

        label_1 = ttk.Label(self, text="This software is supported by the Pacific Northwest National Lab and the DIRECT Program in University of Washington.", background='#ffffff', font='Small_Font')
        label_1.pack()

        photo1 = PhotoImage(file="PNNL.png")
        photo2 = PhotoImage(file="UWDIRECT.png")
        img1 = tk.Label(self, image=photo1, background='#ffffff')
        img2 = tk.Label(self, image=photo2, background='#ffffff')
        img1.image = photo1
        img2.image = photo2
        img1.pack()
        img2.pack()

        label_2= ttk.Label(self, text="The software is created by Ellen Murphy, Xueqiao Zhang, Renlong Zheng.", font='Small_Font', background='#ffffff')
        label_2.pack(pady=40, padx=10)

        button1 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button1.pack()


app = Sea()
app.mainloop()