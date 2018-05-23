import tkinter as tk
import sys
import os
from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as ani

import tkinter.messagebox as tkMessageBox
import itertools
import csv


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
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.tk.call('wm', 'iconphoto', self._w, PhotoImage(file='taiji.png'))
        tk.Tk.wm_title(self, "High-Resolution AFM 3D Visualization Client")
        self.state('zoomed')

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (data_cleaning, load_data, threeD_plot, twoD_slicing, animation, tutorial, acknowledge):
            frame = F (container, self)

            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(data_cleaning)

    def show_frame(self,cont):
        frame = self.frames[cont]
        frame.tkraise()


class data_cleaning(tk.Frame):
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

        z_approach = z_approach.reset_index(drop=True)
        z_retract = z_retractt.reset_index(drop=True)
        return data, z, z_approach, z_retract

    def matconvertor(self):
        #mat = scipy.io.loadmat()
        #phase = mat['PHASEtot']
        return

    def Zsnsr(self):

        return

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')

        label = ttk.Label(self, text="Data Pre-processing", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label_1 = ttk.Label(self, text="Input Dataset",font="Small_Font", background='#ffffff')
        label_1.pack()
        source =ttk.Entry(self)
        source.pack(pady=10, padx=10)

        button0 = ttk.Button(self, text="Get Dataset", command=lambda: self.get_source(source))
        button0.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Leverage",command=lambda: controller.self)
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
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="Input Dataset Information for Visualization", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label_2 = ttk.Label(self, text="x_actual",font="Small_Font", background='#ffffff')
        label_2.pack()
        txtxactual =ttk.Entry(self)
        txtxactual.pack()

        label_3 = ttk.Label(self, text="y_actual",font="Small_Font", background='#ffffff')
        label_3.pack()
        txtyactual =ttk.Entry(self)
        txtyactual.pack()

        label_4 = ttk.Label(self, text="x_size",font="Small_Font", background='#ffffff')
        label_4.pack()
        txtxsize =ttk.Entry(self)
        txtxsize.pack()

        label_5 = ttk.Label(self, text="y_size",font="Small_Font", background='#ffffff')
        label_5.pack()
        txtysize =ttk.Entry(self)
        txtysize.pack()

        button0 = ttk.Button(self, text="Get dataset information", command=lambda: self.get_data(txtxsize, txtysize, txtxactual, txtyactual))
        button0.pack(pady=10, padx=10)

        button1 =ttk.Button(self, text="3D_Plot", command=lambda: controller.show_frame(threeD_plot))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="2D_Slicing", command=lambda: controller.show_frame(twoD_slicing))
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="2D_Slicing_Animation", command=lambda: controller.show_frame(animation))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)


class threeD_plot(tk.Frame):
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
            Z_dir = data.iloc[:,0].iloc[-len(z) // 2:]
            data1 = data.iloc[:,:].iloc[-len(z) // 2:].drop(['Z (nm)'], axis=1)
        else:
            Z_dir = data.iloc[:, 0].iloc[:len(z) // 2]
            data1 = data.iloc[:, :].iloc[:len(z) // 2].drop(['Z (nm)'], axis=1)

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
        im = ax.scatter(xi, yi, zi, c=fxyz, alpha=0.2)
        plt.colorbar(im)
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

        fig.savefig("3D Plot.tif")


    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="3D Plot", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label_1 = ttk.Label(self, text="Z Direction", font="Small_Font", background='#ffffff')
        label_1.pack()
        txtdir = ttk.Entry(self)
        txtdir.pack()

        button0 = ttk.Button(self, text="Get Z Direction",command=lambda: self.Z_direction(txtdir))
        button0.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Get 3D Plot", command=lambda: self.threeDplot(Z_direction, z, x_actual, y_actual, x_size, y_size))
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button2.pack(pady=10, padx=10)

        button3 =ttk.Button(self, text="2D Slicing", command=lambda:controller.show_frame(twoD_slicing))
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)


class twoD_slicing(tk.Frame):
    def Z_direction(self, txtzdir):
        global Z_direction
        global Z_dir
        Z_direction = txtzdir.get()
        if Z_direction == "up":
            Z_dir = z_retract
        else:
            Z_dir = z_approach
        return Z_direction, Z_dir

    def location_slices(self, txtnslices):
        global location_slices
        location_slices = int(txtnslices.get())
        return location_slices

    def export_filename(self, txtfilename):
        global export_filename2
        export_filename2 = txtfilename.get()
        return export_filename2

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):
            phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
            ps = np.array(phaseshift)
            ps_reshape = np.reshape(ps, (x_size, y_size))
            pslist.append(ps_reshape)
        return pslist

    def twoDX_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        global canvas
        global canvas1
        a = np.linspace(init, x_actual, x_size)[location_slices]
        b = np.linspace(init, y_actual, y_size)
        c = Z_dir
        X, Z, Y = np.meshgrid(a, c, b)

        As = np.array(self.create_pslist(x_size, y_size))[init:len(Z_dir), location_slices, :]

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=As, s=6)
        plt.colorbar(im)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('2D X Slicing (X='+str(round(a,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        csvfile = export_filename2+str(".csv")
        # Assuming As is a list of lists
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(As)

        setStr = '{}_Xslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig1 = plt.figure(figsize=(11, 9))
        plt.subplot(111)
        plt.imshow(As, aspect='auto')
        plt.xlabel('Y', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D X Slicing (X='+str(round(a,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)
        plt.colorbar()

        canvas1 = FigureCanvasTkAgg(fig1, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Xslices.tif'.format(export_filename2)
        fig1.savefig(setStr)

    def twoDY_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        global canvas
        global canvas1
        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)[location_slices]
        c = Z_dir
        X, Z, Y = np.meshgrid(a, c, b)

        Bs = np.array(self.create_pslist(x_size, y_size))[init:len(Z_dir), :, location_slices]

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=Bs, s=6)
        plt.colorbar(im)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('2D Y Slicing (Y='+str(round(b,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        csvfile = export_filename2+str(".csv")
        # Assuming res is a list of lists
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(Bs)

        setStr = '{}_Yslices.tif'.format(export_filename2)
        fig.savefig(setStr)

        fig2 = plt.figure(figsize=(11, 9))
        plt.subplot(111)
        plt.imshow(Bs, aspect='auto')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.title('2D Y Slicing (Y='+str(round(b,3)) + 'nm)for the Phase Shift of AFM data', fontsize=13)
        plt.colorbar()

        canvas1 = FigureCanvasTkAgg(fig2, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Yslices.tif'.format(export_filename2)
        fig2.savefig(setStr)

    def twoDZ_slicings(self, location_slices, export_filename2, x_actual, y_actual, x_size, y_size):
        global canvas
        phaseshift = (self.create_pslist(x_size, y_size))[(location_slices)-1]

        a = np.linspace(init, x_actual, x_size)
        b = np.linspace(init, y_actual, y_size)
        X, Z, Y = np.meshgrid(a, Z_dir[location_slices], b)
        l = phaseshift

        fig = plt.figure(figsize=(11, 11))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(X, Y, Z, c=l, s=6)
        plt.colorbar(im)
        ax.set_xlim(left=init, right=x_actual)
        ax.set_ylim(bottom=init, top=y_actual)
        ax.set_zlim(top=Z_dir.max(), bottom=Z_dir.min())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('2D Z Slicing (Z='+str(round(Z_dir[location_slices],3)) + 'nm) for the Phase Shift of AFM data', fontsize=13)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        csvfile = export_filename2+str(".csv")
        # Assuming res is a list of lists
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(phaseshift)

        setStr = '{}_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)


    def twoZ_slicings(self, location_slices, x_actual, y_actual, x_size, y_size):
        global canvas1
        phaseshift = (self.create_pslist(x_size, y_size))[int(location_slices)]

        l = phaseshift

        fig = plt.figure(figsize=(9, 9))
        plt.imshow(l)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('2D Z Slicing (Z='+str(round(Z_dir[location_slices],3)) + 'nm) for the Phase Shift of AFM data', fontsize=13)
        plt.colorbar()

        canvas1 = FigureCanvasTkAgg(fig, self)
        canvas1.show()
        canvas1.get_tk_widget().pack(side=tk.LEFT)

        setStr = '{}_2d_Zslices.tif'.format(export_filename2)
        fig.savefig(setStr)

    def clear(self):
        txtnslices.delete(0, END)
        canvas.get_tk_widget().delete("all")
        canvas1.get_tk_widget().delete("all")

    def __init__(self, parent, controller):
        global txtnslices
        global txtzdir
        global txtfilename
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="2D Slicing", font='Large_Font', background='#ffffff')
        label1.pack()

        label_1 = ttk.Label(self, text="Slices Location", font="Small_Font", background='#ffffff')
        label_1.pack()
        txtnslices = ttk.Entry(self)
        txtnslices.pack()

        label_2 = ttk.Label(self, text="Z Direction", font="Small_Font", background='#ffffff')
        label_2.pack()
        txtzdir = ttk.Entry(self)
        txtzdir.pack()

        label_3 = ttk.Label(self, text="Export Filename", font="Small_Font", background='#ffffff')
        label_3.pack()
        txtfilename = ttk.Entry(self)
        txtfilename.pack()

        button0 = ttk.Button(self, text="Get Location Slices & Directions", command=lambda: (self.Z_direction(txtzdir), self.location_slices(txtnslices), self.export_filename(txtfilename)))
        button0.pack()

        button1 = ttk.Button(self, text="Get 2D X Slicing Plot", command=lambda: self.twoDX_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size))
        button1.pack()

        button2 = ttk.Button(self, text="Get 2D Y Slicing Plot", command=lambda: self.twoDY_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size))
        button2.pack()

        button3 = ttk.Button(self, text="Get 2D Z Slicing Plot", command=lambda: (self.twoDZ_slicings(location_slices, export_filename2, x_actual, y_actual, x_size, y_size), self.twoZ_slicings(location_slices, x_actual, y_actual, x_size, y_size)))
        button3.pack()

        button4 = ttk.Button(self, text="Clear the Inputs", command=lambda: self.clear())
        button4.pack()

        button5 = ttk.Button(self, text="3D Plot", command=lambda: controller.show_frame(threeD_plot))
        button5.pack()

        button6 = ttk.Button(self, text="2D Slicing Animation", command=lambda: controller.show_frame(animation))
        button6.pack()

        button7 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button7.pack()

        button8 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button8.pack()

        label2 = ttk.Label(self, text="The reference level for the plots is set as zero at the substrate surface.", font=(None,10))
        label2.pack()


class animation(tk.Frame):
    def get_animation(self):
        global Z_direction
        global number_slices
        Z_direction = txtzdir.get()
        number_slices = numslices.get()
        return Z_direction, number_slices

    def create_pslist(self, x_size, y_size):
        """The function for reshape the input data file depends on certain shape of the input data file"""
        pslist = []
        for k in range(len(z)):
            phaseshift = data.iloc[k, 1:]  # [from zero row to the end row, from second column to the last column]
            ps = np.array(phaseshift)
            ps_reshape = np.reshape(ps, (x_size, y_size))
            pslist.append(ps_reshape)
        return pslist

    def Z_amination(self, Z_direction, number_slices, x_actual, y_actual, x_size, y_size):
        if Z_direction == "up":
            Z_dir = z_retract
        else:
            Z_dir = z_approach
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(left=init, right=x_size)
        ax.set_ylim(top=y_size, bottom=init)
        ax.set_zlim(bottom=Z_dir.min(), top=Z_dir.max())
        ax.set_xlabel('X(nm)', fontsize=12)
        ax.set_ylabel('Y(nm)', fontsize=12)
        ax.set_zlabel('Z(nm)', fontsize=12)
        ax.set_title('XY Slicing Animation for the AFM Phase Shift', fontsize=15)

        ims = []
        for add in np.arange(int(number_slices)):
            a = np.linspace(init, x_actual, x_size)
            b = np.linspace(init, y_actual, y_size)
            c = Z_dir.iloc[int(add * (len(Z_dir) / int(number_slices)))]
            X, Z, Y = np.meshgrid(a, c, b)

            phaseshift = (self.create_pslist(x_size, y_size))[add]
            l = phaseshift
            ims.append((ax.scatter(X, Y, Z, c=l, s=6)))

        anim = ani.ArtistAnimation(fig, ims)

        canvas1 = tk.Canvas(anim, width=100, height=100)
        canvas1.pack()

    def __init__(self, parent, controller):
        global txtzdir
        global numslices
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label = ttk.Label(self, text="2D Slicing Animation", font='Large_Font', background='#ffffff')
        label.pack(pady=10, padx=10)

        label1 =ttk.Label(self, text="Number of Slices", font='Large_Font', background='#ffffff')
        label1.pack()
        numslices = ttk.Entry(self)
        numslices.pack()

        label2 =ttk.Label(self, text="Z Direction", font='Large_Font', background='#ffffff')
        label2.pack()
        txtzdir = ttk.Entry(self)
        txtzdir.pack()

        label3 =ttk.Label(self, text="2D Slicing Animation", font='Large_Font', background='#ffffff')
        label3.pack()
        entry3 = ttk.Entry(self)
        entry3.pack()

        button1 = ttk.Button(self, text="Get Inputs", command=lambda: self.get_animation())
        button1.pack(pady=10, padx=10)

        button2 = ttk.Button(self, text="Get Z Animation", command=lambda: self.Z_amination(Z_direction, number_slices, x_actual, y_actual, x_size, y_size))
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="Clear the Inputs", command=lambda: controller.clear())
        button3.pack(pady=10, padx=10)

        button4 = ttk.Button(self, text="Organizing Dataset", command=lambda: controller.show_frame(load_data))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(self, text="Home", command=lambda: controller.show_frame(data_cleaning))
        button5.pack(pady=10, padx=10)

class tutorial(ttk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self, background='#ffffff')
        label1 = ttk.Label(self, text="Tutorial", font=Large_Font, background='#ffffff')
        label1.pack()

        label2 = ttk.Label(self, text="Introduction to Our Software", font=Large_Font, background='#ffffff')
        label2.pack()

        def snd1():
            os.system("D:/New/Dropbox/UW/training/Cleanroom/EPFMNMEM2016-V004900_DTH.mp4")

        var = IntVar()

        rb1 = ttk.Button(self, text="Play Video", command=snd1)
        rb1.pack(pady=10, padx=10)

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