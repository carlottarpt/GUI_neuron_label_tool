# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import PySimpleGUI as sg
import h5py

from matplotlib.backends.backend_tkagg import (
 FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib import cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import copy
import tkinter as tk
import cv2
from load_data import *
from PIL import Image, ImageTk
from utils import *

annotation_path_pca_ica = r"C:\Users\carlo\OneDrive - ZHAW\VT\Annotations\dataPackedForGeneration.mat"
# annotation_path_cnmf_e = "/home/user/Documents/single_session/dataPackedForGeneration.mat"
# annotation_path_min1pipe = "/home/user/Documents/single_session/dataPackedForGeneration.mat"
cell_map_pca_ica = r'C:\Users\carlo\OneDrive - ZHAW\VT\Annotations\cellMap.mat'
movie_path = r'U:\single_session\preprocessed\preprocessedMovie.h5'





class App(tk.Frame):
    def __init__(self, window, movie_path,  annotation_path_pca_ica):
        self.window = window

        self.window.configure(background="blue")
        self.window.title("Neuron Labelling")
        self.window.geometry("550x300+300+150")
        self.window.resizable(width=True, height=True)


        self.frame_annotations_1 = tk.Frame(self.window, background="blue")
        self.frame_annotations_1.pack(side=tk.TOP, padx='5', pady='5')
        self.frame_annotations_1.configure(background="cyan")

        self.canvas = tk.Canvas(self.frame_annotations_1, width=400, height=200)
        self.canvas.pack()

        self.frame_buttons_1 = tk.Frame(master=self.window, background = "blue")
        self.frame_buttons_1.pack(side=tk.BOTTOM, padx='5', pady='5')

        self.frame_buttons_2 = tk.Frame(master=self.window)
        self.frame_buttons_2.pack(side=tk.BOTTOM, padx='5', pady='5')

        self.video = Video_Frames(movie_path)
        self.pca_ica = PCAICA(annotation_path_pca_ica)
        self.neuron_pca_ica = None
        self.cellmap_pca_ica = None


        #self.show_annotations()
        self.next_button = tk.Button(self.frame_buttons_1,
                             text="Next", fg="black",
                             command=self.show_annotations)
        self.next_button.pack(side = tk.RIGHT)
        self.PCA_ICA_button = tk.Button(self.frame_buttons_2,
                                     text="PCA-ICA", fg="black",
                                     command=self.window.quit)
        self.PCA_ICA_button.pack(side=tk.LEFT)
        self.CNMFE_button = tk.Button(self.frame_buttons_2,
                             text="CNMF-E", fg="black",
                             command=self.window.quit)
        self.CNMFE_button.pack(side = tk.LEFT)
        self.show_annotations()


    def show_annotations(self):
        self.canvas.delete("all")
        neuron = self.pca_ica.get_neuron()
        cell_map = self.pca_ica.get_cellmap()
        cell_map /= np.max(cell_map)

        img = Image.fromarray(neuron)
        img_cellmap = Image.fromarray(np.uint8(cm.viridis(cell_map)*255))
        #img_cellmap = img_cellmap.rotate
        basewidth = 100
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img_cellmap = img_cellmap.resize((basewidth, hsize), Image.ANTIALIAS)
        self.neuron_pca_ica = ImageTk.PhotoImage(img)
        self.cellmap_pca_ica = ImageTk.PhotoImage(img_cellmap)

        self.canvas.create_image(0, 0, anchor="nw", image=self.neuron_pca_ica)
        self.canvas.create_image(0, 100, anchor="nw", image=self.cellmap_pca_ica)
        #self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

    def on_key_press(self, event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)


class PCAICA():
    def __init__(self, annotation_path_pca_ica):
        self.data = load_annotation_data(annotation_path_pca_ica, "pca_ica")
        self.masks = self.data[0]
        self.cellmap = load_cell_map(cell_map_pca_ica, "pca_ica")
        self.cellmap = self.cellmap.transpose()
        self.counter = 0
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0

    def get_neuron(self):
        self.counter += 1
        neuron, self.x_start, self.x_end, self.y_start, self.y_end = get_cropped_neuron(self.masks[self.counter-1])
        #print((self.x_start, self.x_end, self.y_start, self.y_end))
        #print(self.x_start)
        return neuron

    def get_cellmap(self):
        img = self.cellmap[self.x_start : self.x_end, self.y_start:self.y_end]
        return img



class CNMFE():
    pass
class MIN1PIPE():
    pass

class Video_Frames:
    def __init__(self, video_source=0):

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        # if not self.vid.isOpened():
        #     raise ValueError("Unable to open video source", video_source)

          # Get video source width and height
        # self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):

        if self.vid.isOpened():
            ret, frame = self.vid.read()
        if ret:
             # Return a boolean success flag and the current frame converted to BGR
            return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return (ret, None)


# Create a window and pass it to the Application object
root = tk.Tk()
app = App(root, movie_path, annotation_path_pca_ica)
root.mainloop()



#movie_frames = load_preprocessed_movie(movie_path)

#pca_ica = load_annotation_data(annotation_path_pca_ica, "pca_ica")
#cnmf_e = load_annotation_data(annotation_path_cnmf_e, annotation_type)
#min1pipe = load_annotation_data(annotation_path_min1pipe, annotation_type)

#movie.close()
# if __name__ == '__main__':