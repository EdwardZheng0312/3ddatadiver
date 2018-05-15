import tkinter as tk
import sys
import os
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import animation
import tkinter.messagebox as tkMessageBox
import itertools


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


Large_Font = ("Verdana", 12)
Small_Font = ("Vardana", 6)
init = 0


class Sea(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.tk.call('wm', 'iconphoto', self._w, PhotoImage(file='taiji.png'))
        tk.Tk.wm_title(self, "High-resolution AFM 3D visualization client")

        s = ttk.Style()
        s.configure('My.TFrame', background='red')

        container = ttk.Frame(self, style='My.TFrame')
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (data_cleaning, load_data, threeD_plot, twoD_slicing, animation, acknowledge):
            frame = F (container, self)

            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(data_cleaning)

    def show_frame(self,cont):
        frame = self.frames[cont]
        frame.tkraise()


class data_cleaning(ttk.Frame):
    def get_source(self, source):
        global filename
        global data
        global z
        global z_approach
        global z_retract
        filename = source.get()
        data = self.load_data(filename)[0]
        z = self.load_data(filename)[1]
        z_approach = self.load_data(filename)[2]
        z_retract = self.load_data(filename)[3]
        return filename, data, z, z_approach, z_retract

    def load_data(self, filename):
        """The function to do the primary data clean process on the input data file."""
        data = pd.read_csv(filename)
        z = data.iloc[:, 0]
        z_approach = z[: len(z)//2]
        z_retractt = z[len(z)//2:]
        z_retract = z_retractt.reset_index(drop=True)
        return data, z, z_approach, z_retract

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Data Pre-processing", font='Large_Font')
        label.pack(pady=10, padx=10)

        label_1 = ttk.Label(self, text="Input_Dataset",font="Small_Font")
        label_1.pack()
        source =ttk.Entry(self)
        source.pack()

        button0 = ttk.Button(self, text="Get_Dataset", command=lambda: self.get_source(source))
        button0.pack()

        button1 = ttk.Button(self, text="Leverage",command=lambda: controller.self)
        button1.pack()

        button2 = ttk.Button(self, text="Organizing_Dataset", command=lambda: controller.show_frame(load_data))
        button2.pack()

        button3 = ttk.Button(self, text="Acknowledge", command=lambda: controller.show_frame(acknowledge))
        button3.pack()

        button4 = ttk.Button(self, text="Quit", command=lambda: controller.quit())
        button4.pack()


class load_data(ttk.Frame):
    def get_data(self, txtxsize, txtysize, txtxactual, txtyactual):
        global x_size
        global y_size
        global x_actual
        global y_actual
        x_size = int(txtxsize.get())
        y_size = int(txtysize.get())
        x_actual = int(txtxactual.get())
        y_actual = int(txtyactual.get())
        return data, z, x_size, y_size, x_actual, y_actual

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):
            phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
            ps = np.array(phaseshift)
            ps_reshape = np.reshape(ps, (x_size, y_size))
            pslist.append(ps_reshape)
        return pslist

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Input Dataset Information for Visualization", font='Large_Font')
        label.pack(pady=10, padx=10)

        label_2 = ttk.Label(self, text="x_actual",font="Small_Font")
        label_2.pack()
        txtxactual =ttk.Entry(self)
        txtxactual.pack()

        label_3 = ttk.Label(self, text="y_actual",font="Small_Font")
        label_3.pack()
        txtyactual =ttk.Entry(self)
        txtyactual.pack()

        label_4 = ttk.Label(self, text="x_size",font="Small_Font")
        label_4.pack()
        txtxsize =ttk.Entry(self)
        txtxsize.pack()

        label_5 = ttk.Label(self, text="y_size",font="Small_Font")
        label_5.pack()
        txtysize =ttk.Entry(self)
        txtysize.pack()

        button0 = ttk.Button(self, text="Get dataset information", command=lambda: self.get_data(txtxsize, txtysize, txtxactual, txtyactual))
        button0.pack()

        button1 =ttk.Button(self, text="3D_Plot", command=lambda: controller.show_frame(threeD_plot))
        button1.pack()

        button2 = ttk.Button(self, text="2D_Slicing", command=lambda: controller.show_frame(twoD_slicing))
        button2.pack()

        button3 = ttk.Button(self, text="2D_Slicing_Animation", command=lambda: controller.show_frame(animation))
        button3.pack()

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button4.pack()

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack()


class threeD_plot(ttk.Frame):
    def Z_direction(self, txtdir):
        global Z_direction
        Z_direction = txtdir.get()
        return Z_direction

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):
            phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
            ps = np.array(phaseshift)
            ps_reshape = np.reshape(ps, (x_size, y_size))
            pslist.append(ps_reshape)
        return pslist

    def threeDplot(self, Z_direction, z, x_actual, y_actual, x_size, y_size):
        if Z_direction == "up":
            Z_dir = data.iloc[:,0].iloc[:len(z)//2]
            data1 = data.iloc[:,:].iloc[:len(z)//2].drop(['Z (nm)'], axis=1)
        else:
            Z_dir = data.iloc[:, 0].iloc[-len(z) // 2:]
            data1 = data.iloc[:, :].iloc[-len(z) // 2:].drop(['Z (nm)'], axis=1)

        retract_as_numpy = data1.as_matrix(columns=None)
        retract_as_numpy_reshape1 = retract_as_numpy.reshape(len(Z_dir), x_size, y_size)
        retract_as_numpy_reshape2 = retract_as_numpy_reshape1.flatten('F')

        # Code for the plotting
        x = np.linspace(init, x_actual, x_size)
        y = np.linspace(init, y_actual, y_size)
        z = np.linspace(init, Z_dir.max(), len(Z_dir))

        # This creates a "flat" list representation of a 3Dspace
        points = []
        for element in itertools.product(x, y, z):
            points.append(element)

        fxyz = list(retract_as_numpy_reshape2)
        xi, yi, zi = zip(*points)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xi, yi, zi, c=fxyz, alpha=0.2)
        ax.set_xlim(left=init,right=x_actual)
        ax.set_ylim(top=y_actual, bottom=init)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=15)
        ax.set_ylabel('Y(nm)', fontsize=15)
        ax.set_zlabel('Z(nm)', fontsize=15)
        ax.set_title('3D Plot for Phase Shift of the AFM data', fontsize=20)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="3D_Plot", font='Large_Font')
        label.pack(pady=10, padx=10)

        label_1 = ttk.Label(self, text="Z_direction", font="Small_Font")
        label_1.pack()
        txtdir = ttk.Entry(self)
        txtdir.pack()

        button0 = ttk.Button(self, text="Get Z_direction",command=lambda: self.Z_direction(txtdir))
        button0.pack()

        button1 = ttk.Button(self, text="Get 3D_Plot", command=lambda: self.threeDplot(Z_direction, z, x_actual, y_actual, x_size, y_size))
        button1.pack()

        button2 =ttk.Button(self, text="2D_slicing", command=lambda:controller.show_frame(twoD_slicing))
        button2.pack()

        button3 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button3.pack()

        button4 = ttk.Button(self, text="Organizing_Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack()

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack()


class twoD_slicing(tk.Frame):
    def Z_direction(self, txtzdir):
        global Z_direction
        Z_direction = txtzdir.get()
        return Z_direction

    def location_slices(self, txtnslices, Z_dir):
        global location_slices
        location_slices = (int(txtnslices.get())/(Z_dir.max()))*len(z)
        return location_slices

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):
            phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
            ps = np.array(phaseshift)
            ps_reshape = np.reshape(ps, (x_size, y_size))
            pslist.append(ps_reshape)
        return pslist

    def twoDX_slicings(self, location_slices, Z_direction, x_actual, y_actual, x_size, y_size):
        if Z_direction == "up":
            Z_dir = z_retract
        else:
            Z_dir = z_approach

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)[location_slices]
        X, Z, Y = np.meshgrid(a, Z_dir, b)

        psasas = []
        for k in range(len(z)):
            for i in range(len(a)):
                A = (self.create_pslist(x_size, y_size))[k].transpose()[location_slices][i]
                psasas.append(A)
        l = psasas

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=l, s=6)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=15)
        ax.set_ylabel('Y(nm)', fontsize=15)
        ax.set_zlabel('Z(nm)', fontsize=15)
        ax.set_title('2D Z_Slicing for the Phase Shift of AFM data', fontsize=20)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def twoDY_slicings(self, location_slices, Z_direction, x_actual, y_actual, x_size, y_size):
        if Z_direction == "up":
            Z_dir = z_retract
        else:
            Z_dir = z_approach

        phaseshift = (self.create_pslist(x_size, y_size))[int(location_slices)]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        X, Z, Y = np.meshgrid(a, Z_dir[location_slices], b)
        l = phaseshift

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=l, s=6)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=15)
        ax.set_ylabel('Y(nm)', fontsize=15)
        ax.set_zlabel('Z(nm)', fontsize=15)
        ax.set_title('2D Z_Slicing for the Phase Shift of AFM data', fontsize=20)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def twoDZ_slicings(self, location_slices, Z_direction, x_actual, y_actual, x_size, y_size):
        if Z_direction == "up":
            Z_dir = z_retract
        else:
            Z_dir = z_approach

        phaseshift = (self.create_pslist(x_size, y_size))[int(location_slices)-1]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        X, Z, Y = np.meshgrid(a, Z_dir[location_slices-1], b)
        l = phaseshift

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=l, s=6)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=15)
        ax.set_ylabel('Y(nm)', fontsize=15)
        ax.set_zlabel('Z(nm)', fontsize=15)
        ax.set_title('2D Z_Slicing for the Phase Shift of AFM data', fontsize=20)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="2D_Slicing", font='Large_Font')
        label.pack(pady=10, padx=10)

        lab1 = LabelFrame(self)
        lab1_2 = Listbox(lab1, exportselection=0)
        lab1_2.insert(1, "X_Slicing")
        lab1_2.insert(2, "Y_Slicing")
        lab1_2.insert(3, "Z_Slicing")

        label_1 = ttk.Label(self, text="Location_Slices", font="Small_Font")
        label_1.pack()
        txtlocslices = ttk.Entry(self)
        txtlocslices.pack()

        label_2 = ttk.Label(self, text="Z_Direction", font="Small_Font")
        label_2.pack()
        txtzdir = ttk.Entry(self)
        txtzdir.pack()

        button0 = ttk.Button(self, text="Get Location_Slices & Directions", command=lambda: (self.location_slices(txtnslices, Z_dir), self.Z_direction(txtzdir)))
        button0.pack()

        button1 = ttk.Button(self, text="Get 2D X_Slicing Plot", command=lambda: self.twoDY_slicings(location_slices, Z_direction, x_actual, y_actual, x_size, y_size))
        button1.pack()

        button2 = ttk.Button(self, text="Get 2D Y_Slicing Plot", command=lambda: self.twoDY_slicings(location_slices, Z_direction, x_actual, y_actual, x_size, y_size))
        button2.pack()

        button3 = ttk.Button(self, text="Get 2D Z_Slicing Plot", command=lambda: self.twoDZ_slicings(location_slices, Z_direction, x_actual, y_actual, x_size, y_size))
        button3.pack()

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button4.pack()

        button5 = ttk.Button(self, text="3D_Plot", command=lambda: controller.show_frame(threeD_plot))
        button5.pack()

        button6 = ttk.Button(self, text="2D_Slicing_Animation", command=lambda: controller.show_frame(animation))
        button6.pack()

        button7 = ttk.Button(self, text="Organizing_Dataset", command=lambda: controller.show_frame(load_data))
        button7.pack()

        button8 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button8.pack()

class animation(tk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="2D_Slicing_Animation", font='Large_Font')
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button1.pack()

        button2 = ttk.Button(self, text="Organizing_Dataset", command=lambda: controller.show_frame(load_data))
        button2.pack()

        button3 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(load_data))
        button3.pack()


class acknowledge(tk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Acknowledge", font='Large_Font')
        label.pack(pady=10, padx=10)

        label_1 = ttk.Label(self, text="This software is supported by the Pacific Northwest National Lab and the DIRECT Program in University of Washington.", font='Small_Font')
        label_1.pack()

        photo1 = PhotoImage(file="PNNL.png")
        photo2 = PhotoImage(file="UWDIRECT.png")
        img1 = tk.Label(self, image=photo1)
        img2 = tk.Label(self, image=photo2)
        img1.image = photo1
        img2.image = photo2
        img1.pack()
        img2.pack()

        label_2= ttk.Label(self, text="The software is created by Ellen Murphy, Xueqiao Zhang, Renlong Zheng.", font='Small_Font')
        label_2.pack(pady=40, padx=10)

        button1 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button1.pack()


app = Sea()
app.mainloop()
