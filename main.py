from PyQt5.QtCore import  Qt, QUrl, QPointF
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import  QGraphicsVideoItem
from load_data import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import cm
from scipy.io import savemat
from PIL import Image, ImageDraw
import math
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from scipy.signal import find_peaks
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from enum import Enum
from functools import partial
from PyQt5 import QtWidgets, QtGui, QtCore


######################### set paths here ################################################################
synth = False
if not synth:
    #pca-ica files
    annotation_path_pca_ica = r"/home/user/Documents/single_session/dataPackedForGeneration.mat"
    cell_map_pca_ica_path = r'/home/user/Documents/single_session/extracted/cellMap.mat'
    annotation_path2_pca_ica = r"/home/user/Documents/single_session/extracted/resultsPCAICA.mat"

    #movie
    movie_path = r'/home/user/Documents/single_session/preprocessed/preprocessedMovie5.avi'
    movie_path_hdf5 = r'/home/user/Documents/single_session/preprocessed/preprocessedMovie.h5'
    #cnmfe
    cnmfe_masks_weighted_path = "/home/user/Documents/MATLAB/utils/binary_masks.mat"
    cnmfe_cellmap_path = "/home/user/Documents/MATLAB/utils/cellmap.mat"
    cnmfe_traces_path = '/home/user/Documents/MATLAB/utils/traces.mat'
    cnmfe_coords_path = ""

    #save
    outfile_path = r'/home/user/Documents/single_session/GUI/singleshot/'

else:
    movie_path = r'/home/user/Documents/synthetic_datasets/sequence_10/video/video.avi'
    movie_path_hdf5 = r'/home/user/Documents/synthetic_datasets/sequence_10/video/movie.h5'

    cnmfe_masks_weighted_path = '/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/binary_masks_synthetic.mat'
    cnmfe_cellmap_path = '/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/cellmap_synthetic.mat'
    cnmfe_traces_path = '/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/traces_synthetic.mat'
    cnmfe_coords_path = '/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/coords_synthetic.mat'

    path_filters_binary = r'/home/user/Documents/synthetic_datasets/sequence_10/extracted/filters_binary.mat'
    annotation_path2_pca_ica = r'/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsPCAICA/resultsPCAICA.mat'
    cell_map_pca_ica_path = r'/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsPCAICA/cellMap.mat'
    # save
    outfile_path = r'/home/user/Documents/single_session/GUI/synthetic/'

if not os.path.exists(outfile_path):
    os.makedirs(outfile_path)
####################### settings ########################################################################
window_size = (640, 480)            #window size (rescalable during program runtime)
polygon_is_movable = True           #Do you want your Annotation Shape to be movable?
polygon_point_is_movable = True     #Do you want single points of your Annotation Shape to be movable?
colormap1 = 'Purples'               #Colormap for PCA-ICA Annotations
colormap2 = 'Greens'                #Colormap for CNMFE Annotations
colorfade = True                    #dark to light --> SNR best to worst

if not os.path.exists(outfile_path):
    os.makedirs(outfile_path)

scale_factor = 0
center = (0,0)
scale_factor_movie = 0

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, title="calcium_traces"):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # fig, axes = plt.subplots(2, )
        self.axes = fig.add_subplot(111)
        self.axes.set_title(title)
        super(MplCanvas, self).__init__(fig)

class VideoPlayer(QtWidgets.QMainWindow):
    """
    Video Player Window
    """
    def __init__(self, main_scene, parent=None
                 ):
        super(VideoPlayer, self).__init__(parent)
        self.view = AnnotationView()
        self.scene = QtWidgets.QGraphicsScene()
        self.view.setScene(self.scene)
        self.main_scene = main_scene
        video_item = QGraphicsVideoItem()
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.mediaPlayer.setVideoOutput(video_item)
        self.mediaPlayer.setMedia(
            QMediaContent(QUrl.fromLocalFile(movie_path)))
        self.view.scene().addItem(video_item)
        self.setCentralWidget(self.view)
        self.setWindowTitle("Movie")
        self.view.fitInView(video_item, QtCore.Qt.KeepAspectRatio)

        video_item.setSize(QtCore.QSizeF(500,500))
        self.mediaPlayer.play()
        self.view.fitInView(video_item, QtCore.Qt.KeepAspectRatio)
        self.polygons_dict={"pca_ica":[], "cnmfe": []}

        show_menu = self.menuBar().addMenu("Show")
        show_pca_ica_action = show_menu.addAction("PCA-ICA")
        show_pca_ica_action.triggered.connect(partial(self.load_annotations, "pca_ica"))
        show_cnmfe_action = show_menu.addAction("CNMF-E")
        show_cnmfe_action.triggered.connect(partial(self.load_annotations, "cnmfe"))

        hide_menu = self.menuBar().addMenu("Hide")
        hide_pca_ica_action = hide_menu.addAction("PCA-ICA")
        hide_pca_ica_action.triggered.connect(partial(self.hide_annotations, "pca_ica"))
        hide_cnmfe_action = hide_menu.addAction("CNMF-E")
        hide_cnmfe_action.triggered.connect(partial(self.hide_annotations, "cnmfe"))

    def load_annotations(self, key):
        if key == "pca_ica":
            color = "purple"
        else:
            color = "green"
        coords_dict, traces_dict, polygons_dict = self.main_scene.get_current_annotations()

        for coord in coords_dict[key]:
            qpoints = []
            for x, y in coord:
                qpoints.append(QtCore.QPointF(x,y))
            qpolygon = (QtGui.QPolygonF(qpoints))

            qgraphicspolygon = QtWidgets.QGraphicsPolygonItem(qpolygon)
            qgraphicspolygon.setPen(QtGui.QPen(QtGui.QColor(color), 1))
            self.polygons_dict[key].append(qgraphicspolygon)
            self.scene.addItem(qgraphicspolygon)

    def hide_annotations(self, key):
        for poly in self.polygons_dict[key]:
            self.scene.removeItem(poly)
            del poly
        self.polygons_dict[key] = []


class TraceWindow(QtWidgets.QMainWindow):
    """
    Trace Window showing calcium traces of selected neuron ROI
    """
    def __init__(self, main_scene, parent=None):
        super(TraceWindow, self).__init__(parent)

        self.main_scene = main_scene
        self.start_value = None
        self.end_value = None
        self.setWindowTitle("Traces")

        self.traces_dict = {"pca_ica":[], "cnmfe": []}

        self.form_widget = FormWidget(self)
        self.UiComponents()
        self.setCentralWidget(self.form_widget)



    def show_traces(self, keys=("pca_ica", "cnmfe", "added")):
        for i in reversed(range(self.form_widget.layout.count())):
            self.form_widget.layout.itemAt(i).widget().setParent(None)

        self.traces_dict = {"pca_ica": [], "cnmfe": [], "added": []}
        coords_dict, traces_dict, polygons_dict = self.main_scene.get_current_annotations()

        for key in keys:

            for i, poly in enumerate(polygons_dict[key]):
                if poly.isSelected():
                    self.traces_dict[key].append(traces_dict[key][i])
                    self.plot_trace(traces_dict[key][i], key)

    def UiComponents(self):

        # creating a QLineEdit object
        self.line_edit = QLineEdit("start_frame", self)

        # setting geometry
        self.line_edit.setGeometry(250, 0, 100, 30)

        # creating a QLineEdit object
        self.line_edit_end = QLineEdit("end_frame", self)

        # setting geometry
        self.line_edit_end.setGeometry(370, 0, 100, 30)

        # adding action to the line edit when enter key is pressed
        self.line_edit.returnPressed.connect(lambda: self.do_action())
        self.line_edit_end.returnPressed.connect(lambda: self.do_action_end())

        # method to do action
    def do_action(self):
        # getting text from the line edit
        self.start_value = int(self.line_edit.text())

    def do_action_end(self):
        # getting text from the line edit
        self.end_value = int(self.line_edit_end.text())


    def plot_trace(self, trace, key, xlabel="Frame #", ylabel="Intensity (norm.)", sharex=False):
        plot = MplCanvas(self, width=40, height=4, dpi=100, title=key)
        if not self.start_value and not self.end_value:
            frames = np.arange(len(trace.transpose()))
            plot.axes.plot(frames, trace.transpose())
        else:
            frames = np.arange(self.end_value - self.start_value)
            plot.axes.plot(frames, list(trace.transpose())[self.start_value:self.end_value])

        plot.axes.set_xlabel(xlabel)
        plot.axes.set_ylabel(ylabel)
        self.form_widget.layout.addWidget(plot)


class GripItem(QtWidgets.QGraphicsPathItem):

    def __init__(self, annotation_item, index, color="green", circle_size=1, square_size=2, circle=True, z_value=10):
        super(GripItem, self).__init__()
        self.color = color
        self.circle_size = circle_size
        self.square_size = square_size
        self.z_value = z_value
        self.circle = QtGui.QPainterPath()
        self.circle.addEllipse(QtCore.QRectF(- self.circle_size / 2, - self.circle_size / 2,
                                             self.circle_size, self.circle_size))
        self.square = QtGui.QPainterPath()
        self.square.addRect(QtCore.QRectF(- self.square_size / 2, - self.square_size / 2,
                                          self.square_size, self.square_size))

        self.m_annotation_item = annotation_item
        self.m_index = index

        if circle:
            self.setPath(self.circle)

        self.setBrush(QtGui.QColor(self.color))
        self.setPen(QtGui.QPen(QtGui.QColor(self.color), circle_size))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, polygon_point_is_movable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(self.z_value)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def hoverEnterEvent(self, event):
        self.setPath(self.square)
        self.setBrush(QtGui.QColor("red"))
        super(GripItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPath(self.circle)
        self.setBrush(QtGui.QColor(self.color))
        super(GripItem, self).hoverLeaveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setSelected(False)
        super(GripItem, self).mouseReleaseEvent(event)
        # update_trace()
        # update value!!

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            # and self.isEnabled():
            self.m_annotation_item.movePoint(self.m_index, value)
            #print("item position has changed", change, value)
        return super(GripItem, self).itemChange(change, value)


class PolygonAnnotation(QtWidgets.QGraphicsPolygonItem):
    '''
    Neuron ROI class
    '''
    def __init__(self, parent=None, movable=polygon_is_movable, selectable=True, color = "green"):
        super(PolygonAnnotation, self).__init__(parent)
        self.m_points = []
        self.color = color
        self.setZValue(10)
        self.z_value = 10
        if type(self.color) != str:
            self.setPen(QtGui.QPen(QtGui.QColor(self.color[0], self.color[1], self.color[2]), 1))
        else:
            self.setPen(QtGui.QPen(QtGui.QColor(self.color), 2))
        self.setAcceptHoverEvents(True)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, selectable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, movable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # m_items seems to hold different coords once the item has been moved
        # (something related to relative distances to parent) but it is crucial for mapping the polygon correctly to the scene.
        # Therefore, a new instance m_coords is created for the corrrect extraction of the labels.
        self.m_items = []
        self.m_coords = []

    def number_of_points(self):
        return len(self.m_items)

    def addPoint(self, p, color="green", circle_size=2, square_size=4):
        self.m_points.append(p)
        self.m_coords.append(p)
        self.setPolygon(QtGui.QPolygonF(self.m_points))
        item = GripItem(self, len(self.m_points) - 1, color = color,
                        circle_size=circle_size, square_size=square_size, z_value=self.z_value)
        self.scene().addItem(item)
        self.m_items.append(item)
        item.setPos(p)

    def removeLastPoint(self):
        if self.m_points:
            self.m_points.pop()
            self.m_coords.pop()
            self.setPolygon(QtGui.QPolygonF(self.m_points))
            it = self.m_items.pop()

            self.scene().removeItem(it)
            del it

    def removeAll(self):
        self.setPolygon(QtGui.QPolygonF([]))
        for point, coord, item in zip(self.m_points, self.m_coords, self.m_items):
            self.scene().removeItem(item)
            del point
            del item
            del coord
        self.m_points = []
        self.m_coords = []
        self.m_items = []

    def movePoint(self, i, p):
        if 0 <= i < len(self.m_points):
            #print("mv pt", p)
            self.m_coords[i] = p
            self.m_points[i] = self.mapFromScene(p)
            #print("mv pt post", self.m_points[i])
            self.setPolygon(QtGui.QPolygonF(self.m_points))

    def move_item(self, index, pos):
        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            #print("I was moved dumbass", change, value)
            for i, point in enumerate(self.m_points):
                self.move_item(i, self.mapToScene(point))

        return super(PolygonAnnotation, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.setBrush(QtGui.QColor(255, 0, 0, 100))
        super(PolygonAnnotation, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        super(PolygonAnnotation, self).hoverLeaveEvent(event)


# classes to manage drawing and hide/show instructions
class Instructions_Draw(Enum):
    No_Instruction = 0
    Polygon_Instruction = 1

class Instructions_Annotations_PCA_ICA(Enum):
    Hide_Instruction = 0
    Show_Instruction = 1

class Instructions_Annotations_CNMFE(Enum):
    Hide_Instruction = 0
    Show_Instruction = 1



class AnnotationScene(QtWidgets.QGraphicsScene):
    """
    Annotation scene is the manager of the complete main scene and holds all polygon objects, annotations etc
    """
    def __init__(self, parent=None):
        super(AnnotationScene, self).__init__(parent)

        self.polygon_item = None

        if not synth:
            self.pca_ica_annotations = load_annotation_data(annotation_path_pca_ica, "pca_ica")
            self.pca_ica_masks = self.pca_ica_annotations[0]
        else:
            self.pca_ica_masks_mat = scipy.io.loadmat(path_filters_binary)
            self.pca_ica_masks = self.pca_ica_masks_mat["filtersBinary"].T

        self.cnmfe_masks = None
        self.movie = load_preprocessed_movie(movie_path_hdf5)
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.addItem(self.image_item)

        self.current_draw_instruction = Instructions_Draw.No_Instruction
        self.current_pca_ica_instruction = Instructions_Annotations_PCA_ICA.Hide_Instruction
        self.current_cnmfe_instruction = Instructions_Annotations_CNMFE.Hide_Instruction


        self.polygons_dict = {"pca_ica": [], "cnmfe": [], "added": []}
        self.coords_dict = {"pca_ica": [], "cnmfe": [], "added": []}
        self.traces_dict = {"pca_ica": scipy.io.loadmat(annotation_path2_pca_ica)["traces"], "cnmfe": scipy.io.loadmat(cnmfe_traces_path)["C"], "added": []}
        self.cellmap_dict = {"pca_ica": load_cell_map(cell_map_pca_ica_path, "pca_ica"),
                             "cnmfe": load_cell_map(cnmfe_cellmap_path, "cnmf-e")}
        if not synth:
            self.events_dict = {"pca_ica": self.pca_ica_annotations[1], "cnmfe": [], "added": []}
        self.export_dict = {"coordinates": [], "binary_masks": [], "traces": [], "events": [],
                            "cellmaps": self.cellmap_dict}


    def get_end_coordinates(self, polygon_item):
        return(polygon_item.mapFromScene(polygon_item.m_points[0]))

    def get_current_annotations(self):
        return self.coords_dict, self.traces_dict, self.polygons_dict

    def load_image(self, filename):
        self.image_item.setPixmap(QtGui.QPixmap(filename))
        self.setSceneRect(self.image_item.boundingRect())

    def setCurrentDrawInstruction(self, instruction):

        self.current_draw_instruction = instruction
        if self.current_draw_instruction == Instructions_Draw.Polygon_Instruction:
            self.polygon_item = PolygonAnnotation()
            self.addItem(self.polygon_item)
            self.polygons_dict["added"].append(self.polygon_item)
        else:
            trace = self.get_neuron_trace()
            self.traces_dict["added"].append(np.array(trace))

    def get_neuron_trace(self):
        coord_neuron = []
        for point in self.polygon_item.m_coords:
            x = point.x()
            y = point.y()
            coord_neuron.append([x,y])

        polygon = np.array(coord_neuron)
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        polygon = tuple(map(tuple, polygon))
        width = self.movie.shape[1]
        height = self.movie.shape[2]

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)
        mask = mask == 1

        trace = [np.mean(self.movie[i][np.transpose(mask)]) for i in range(self.movie.shape[0])]

        return trace


    def load_annotations(self, key):

        if key == "pca_ica":
            self.coords_dict[key], index_of_del_masks = get_coord_from_binary(self.pca_ica_masks)
            self.polygons_dict[key] = (self.plot_annotations(self.coords_dict[key]))

            for i in index_of_del_masks:
                self.traces_dict[key] = np.delete(self.traces_dict[key], i, axis=0)
                if not synth:
                    self.events_dict[key] = np.delete(self.events_dict[key], i, axis=0)

        if key == "cnmfe":
            self.cnmfe_masks = apply_two_third_filter(cnmfe_masks_weighted_path)
            self.coords_dict[key], index_of_del_masks = get_coord_from_binary(self.cnmfe_masks.astype(bool))
            # bigger coord outlines of masks without 2/3 threshold
            #self.coords_dict[key] = load_cnmfe_coord(cnmfe_coords_path)
            self.polygons_dict[key] = (self.plot_annotations(self.coords_dict[key], colormap=colormap2))

            for i in index_of_del_masks:
                self.traces_dict[key] = np.delete(self.traces_dict[key], i, axis=0)

    def hide_annotations(self, key):
        for poly in self.polygons_dict[key]:
            poly.hide()
            for item in poly.m_items:
                item.hide()

    def show_annotations(self, key):
        for poly in self.polygons_dict[key]:
            poly.show()
            for item in poly.m_items:
                item.show()

    def setCurrentPCAICAShowInstruction(self, instruction):
        # instruction can either be Show_Instruction (1) or Hide_Instruction (0)
        self.current_pca_ica_instruction = instruction

        if self.current_pca_ica_instruction == Instructions_Annotations_PCA_ICA.Hide_Instruction:
            self.hide_annotations("pca_ica")
        else:
            self.show_annotations("pca_ica")

    def setCurrentCNMFEShowInstruction(self, instruction):
        # instruction can either be Show_Instruction (1) or Hide_Instruction (0)
        self.current_cnmfe_instruction = instruction

        if self.current_cnmfe_instruction == Instructions_Annotations_CNMFE.Hide_Instruction:
            self.hide_annotations("cnmfe")
        else:
            self.show_annotations("cnmfe")

    def set_z_value(self, value='bottom', keys=('pca_ica','cnmfe','added')):

        # if not isinstance(value, int):
        #     raise ValueError('Zvalue must be int or "bottom" or "top"')
        for key in keys:

            for poly in self.polygons_dict[key]:
                if poly.isSelected():
                    if value == "bottom":
                        value = poly.z_value - 1
                        poly.z_value -= 1
                        value_point = poly.m_items[0].z_value -1
                        for item in poly.m_items:
                            item.z_value -= 1
                            item.setZValue(value_point)
                    if value == "top":
                        value = poly.z_value + 1
                        poly.z_value += 1
                        value_point = poly.m_items[0].z_value + 1
                        for item in poly.m_items:
                            item.z_value += 1
                            item.setZValue(value_point)

                    poly.setZValue(value)

    def delete_polygon(self, keys=("pca_ica","cnmfe","added")):

        for key in keys:

            for i, poly in enumerate(self.polygons_dict[key]):
                if poly.isSelected():
                    poly.removeAll()
                    self.polygons_dict[key].remove(poly)
                    self.traces_dict[key] = np.delete(self.traces_dict[key], i, axis=0)
                    if not synth:
                        if key == "pca_ica":
                            self.events_dict[key] = np.delete(self.events_dict[key], i, axis=0)
                    if key!= "added":
                        self.coords_dict[key] = np.delete(self.coords_dict[key], i, axis=0)

    def coords2mask(self, coords):

        width = 500
        height = 500

        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(coords, outline=1, fill=1)
        mask = np.array(img)
        return mask

    def spike_finder(self, trace, thresh=0.7, distance=1):


        maximum = np.max(np.array(trace))
        min_height = thresh * maximum
        spikes = find_peaks(x=trace, height=min_height, distance=distance)
        return spikes[0].tolist()

    def export2Matfile(self, keys=["pca_ica", "cnmfe", "added"]):
        final_coords = []
        final_binary_masks = []
        final_traces = []
        final_events = []
        for key in keys:

            for i, (poly, trace) in enumerate(zip(self.polygons_dict[key], self.traces_dict[key])):
                final_coords_single = []

                for point in poly.m_coords:
                    #print("pt", point)
                    #print("scene", poly.mapFromScene(point))
                    x = point.x()
                    y = point.y()
                    final_coords_single.append((x,y))

                if final_coords_single:
                    final_coords.append(final_coords_single)
                    mask = self.coords2mask(final_coords_single)
                    final_binary_masks.append(mask)
                    final_traces.append(trace)
                    #print(self.spike_finder(trace))
                    if synth:
                        if key == "cnmfe":
                            final_events.append(self.spike_finder(trace, thresh=0.1))
                        else:
                            final_events.append(self.spike_finder(trace, thresh=0.7))
                    else:
                        if key == "cnmfe":
                            final_events.append(self.spike_finder(trace, thresh=0.1))
                            #print("cnmfe:",self.spike_finder(trace, thresh=0.25))
                        if key == "pca_ica":
                            #print("pca_ica",self.events_dict[key][0])
                            final_events.append(self.events_dict[key][i])
                        if key == "added":
                            final_events.append(self.spike_finder(trace, thresh=0.6))
        print(len(final_traces))
        print(len(final_events))
        self.export_dict["coordinates"] = final_coords
        self.export_dict["binary_masks"] = final_binary_masks
        self.export_dict["traces"] = final_traces
        self.export_dict["events"] = final_events

        savemat(outfile_path + "gui_extracted.mat", self.export_dict)
        print("Saved to " + outfile_path + "gui_extracted.mat !")

    def mousePressEvent(self, event):
        if self.current_draw_instruction == Instructions_Draw.Polygon_Instruction:
            self.polygon_item.removeLastPoint()
            self.polygon_item.addPoint(event.scenePos())
            # movable element
            self.polygon_item.addPoint(event.scenePos())
        super(AnnotationScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.current_draw_instruction == Instructions_Draw.Polygon_Instruction:
            self.polygon_item.movePoint(self.polygon_item.number_of_points()-1, event.scenePos())

        if (event.buttons() and Qt.LeftButton):
            pass
            # print(self.get_end_coordinates(self.polygons_dict["added"][0]))
            #QGraphicsItem.mouseMoveEvent(event)
            #print(self.mapFromScene(self))
            # print(self.pos().x())
            # print(self.pos().y())


        super(AnnotationScene, self).mouseMoveEvent(event)

    def plot_annotations(self, xycoords, movable=polygon_is_movable, selectable=True, colormap=colormap1):

        polygons = []

        col = MplColorHelper(colormap, 0, len(xycoords))

        i = len(xycoords)
        for xycoords_single in xycoords:
            if len(xycoords_single) is 0:
                print("empty")
            if colorfade:
                c = col.get_rgb(i)
                c = [val * 255 for val in c]
            else:
                c = (0,255,0)
            poly = PolygonAnnotation(movable=movable, selectable=selectable, color=c)
            self.addItem(poly)
            for x, y in xycoords_single:
                poly.addPoint(QPointF(x, y),
                              color=QtGui.QColor(c[0], c[1], c[2], 255),
                              circle_size=1, square_size=2)

            polygons.append(poly)
            i = i - 1

        return polygons


class AnnotationView(QtWidgets.QGraphicsView):

    """
    View handles all 'camera' features such as zoom and mappings of the scene
    """

    factor = 2.0

    def __init__(self, parent=None):
        super(AnnotationView, self).__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)


        QtWidgets.QShortcut(QtGui.QKeySequence.ZoomIn, self, activated=self.zoomIn)
        QtWidgets.QShortcut(QtGui.QKeySequence.ZoomOut, self, activated=self.zoomOut)

    def get_center(self):
        return self.mapToScene(self.viewport().rect().center())

    @QtCore.pyqtSlot()
    def zoomIn(self):
        global scale_factor
        global scale_factor_movie
        self.zoom(AnnotationView.factor)

        scale_factor = scale_factor + 1


    @QtCore.pyqtSlot()
    def zoomOut(self):
        global scale_factor
        global scale_factor_movie

        self.zoom(1 / AnnotationView.factor)

        scale_factor = scale_factor - 1


    def zoom(self, f):
        self.scale(f, f)

    def zoom_beginning(self):
        global scale_factor


        if scale_factor < 0:

            self.zoom(1 / (AnnotationView.factor * scale_factor))

        if scale_factor > 0:

            self.zoom(AnnotationView.factor * scale_factor)

    def update_movie_zoom(self):
        global scale_factor
        global scale_factor_movie

        if scale_factor - scale_factor_movie < 0:
            i=1
            for i in range(abs(scale_factor - scale_factor_movie)):
                self.zoom(1 / (AnnotationView.factor))
            scale_factor_movie -= abs(scale_factor - scale_factor_movie)
        if scale_factor - scale_factor_movie > 0:
            i=1
            for i in range(abs(scale_factor - scale_factor_movie)):
                self.zoom(AnnotationView.factor)
            scale_factor_movie += abs(scale_factor - scale_factor_movie)

    def zoomMovieBeginning(self):
        self.zoom((AnnotationView.factor))


class AnnotationWindow(QtWidgets.QMainWindow):
    """
    Main Window
    """
    def __init__(self, parent=None):
        super(AnnotationWindow, self).__init__(parent)

        self.m_view = AnnotationView()
        self.m_scene = AnnotationScene(self)
        self.m_view.setScene(self.m_scene)
        self.setCentralWidget(self.m_view)
        self.w = None

        self.title = "Neuron Labelling"
        self.setWindowTitle(self.title)

        self.cellmap_type = "pca_ica"
        self.video_window = None
        self.trace_window = None

        self.show_annotations = {"pca_ica": False, "cnmfe": False}
        self.loaded_annotations = {"pca_ica": False, "cnmfe": False}

        self.save_cellmaps()
        self.load_cellmap()
        self.create_menus()


        #keybord short cuts
        QtWidgets.QShortcut(QtCore.Qt.Key_Escape, self, activated=partial(self.m_scene.setCurrentDrawInstruction, Instructions_Draw.No_Instruction))
        QtWidgets.QShortcut(QtCore.Qt.Key_Delete, self,activated=self.delete_polygons)
        QtWidgets.QShortcut(QtCore.Qt.Key_P, self,activated=self.load_pca_ica_cellmap)
        QtWidgets.QShortcut(QtCore.Qt.Key_C, self, activated=self.load_cnmfe_cellmap)
        QtWidgets.QShortcut(QtCore.Qt.Key_B, self, activated=self.set_z_value)
        QtWidgets.QShortcut(QtCore.Qt.Key_T, self, activated=partial(self.set_z_value, value="top"))
        QtWidgets.QShortcut(QtCore.Qt.Key_V, self, activated=self.update_movie_zoom)
        QtWidgets.QShortcut(QtCore.Qt.Key_U, self, activated=self.update_traces)
        QtWidgets.QShortcut(QtCore.Qt.Key_D, self, activated=partial(self.m_scene.setCurrentDrawInstruction, Instructions_Draw.Polygon_Instruction))


    def save_cellmaps(self):
        cellmap_pca_ica = load_cell_map(cell_map_pca_ica_path ,"pca_ica")
        cellmap_cnmfe = load_cell_map(cnmfe_cellmap_path, "cnmf-e")
        im_1 = Image.fromarray(cellmap_pca_ica/np.percentile(cellmap_pca_ica, 99.99)*255)
        if im_1.mode != 'RGB':
            im_1 = im_1.convert('RGB')
        im_1.save('cellmap_pca_ica.jpg')
        im_2 = Image.fromarray(cellmap_cnmfe*255)
        if im_2.mode != 'RGB':
            im_2 = im_2.convert('RGB')
        im_2.save('cellmap_cnmfe.jpg')

    def set_z_value(self, value="bottom"):
        self.m_scene.set_z_value(value)

    def delete_polygons(self):
        self.m_scene.delete_polygon()

    def create_menus(self):
        menu_cellmap = self.menuBar().addMenu("Cell Map")
        cellmap_pca_ica_action = menu_cellmap.addAction("PCA-ICA")
        cellmap_pca_ica_action.triggered.connect(self.load_pca_ica_cellmap)

        cnmfe_action = menu_cellmap.addAction("CNMF-E")
        cnmfe_action.triggered.connect(self.load_cnmfe_cellmap)

        menu_annotation = self.menuBar().addMenu("Annotations")
        annotation_pca_ica_action = menu_annotation.addAction("Show/Hide PCA-ICA")
        annotation_pca_ica_action.triggered.connect(self.show_hide_pca_ica_annotations)
        annotation_cnmfe_action = menu_annotation.addAction("Show/Hide CNMF-E")
        annotation_cnmfe_action.triggered.connect(self.show_hide_cnmfe_annotations)

        menu_instructions = self.menuBar().addMenu("Edit Neurons")
        polygon_action = menu_instructions.addAction("Draw new")
        polygon_action.triggered.connect(partial(self.m_scene.setCurrentDrawInstruction, Instructions_Draw.Polygon_Instruction))

        menu_show = self.menuBar().addMenu("Show")
        movie_action = menu_show.addAction("Movie")
        movie_action.triggered.connect(self.show_new_video_window)
        trace_action = menu_show.addAction("Traces")
        trace_action.triggered.connect(self.show_new_trace_window)

        menu_save = self.menuBar().addMenu("Export")
        matfile_action = menu_save.addAction("Matfile")
        matfile_action.triggered.connect(self.exportMatfile)


    def show_new_video_window(self):

        self.video_window = VideoPlayer(self.m_scene)
        self.video_window.resize(window_size[0], window_size[1])
        self.video_window.show()

    def update_movie_zoom(self):
        if self.video_window:
            global scale_factor
            global scale_factor_movie

            center = self.m_view.get_center()
            self.video_window.view.centerOn(center)
            center2 = self.video_window.view.get_center()
            self.video_window.view.update_movie_zoom()

    def show_new_trace_window(self):

        self.trace_window = TraceWindow(self.m_scene)
        self.trace_window.resize(700,500)
        self.trace_window.show()

    def update_traces(self):
        if self.trace_window:
            self.trace_window.show_traces()

    def load_cellmap(self):
        global scale_factor
        global center
        if self.cellmap_type == "pca_ica":
            center = self.m_view.get_center()
            self.m_scene.load_image('cellmap_pca_ica.jpg')
            self.m_view.fitInView(self.m_scene.image_item, QtCore.Qt.KeepAspectRatio)

            self.m_view.centerOn(center)
            if scale_factor != 0:
                self.m_view.zoom_beginning()

        if self.cellmap_type == "cnmf-e":
            center = self.m_view.get_center()
            self.m_scene.load_image('cellmap_cnmfe.jpg')
            self.m_view.fitInView(self.m_scene.image_item, QtCore.Qt.KeepAspectRatio)

            self.m_view.centerOn(center)
            if scale_factor != 0:
                self.m_view.zoom_beginning()

    @QtCore.pyqtSlot()
    def load_pca_ica_cellmap(self):
        if self.cellmap_type == "pca_ica":
            pass
        if self.cellmap_type == "cnmf-e":
            self.cellmap_type = "pca_ica"
            self.load_cellmap()

    @QtCore.pyqtSlot()
    def load_cnmfe_cellmap(self):
        if self.cellmap_type == "pca_ica":
            self.cellmap_type = "cnmf-e"
            self.load_cellmap()
        if self.cellmap_type == "cnmf-e":
            pass

    @QtCore.pyqtSlot()
    def show_hide_pca_ica_annotations(self):

        if self.loaded_annotations["pca_ica"]:
            if self.show_annotations["pca_ica"]:
                self.show_annotations["pca_ica"] = False
                self.m_scene.setCurrentPCAICAShowInstruction(Instructions_Annotations_PCA_ICA.Hide_Instruction)
            else:
                self.show_annotations["pca_ica"] = True
                self.m_scene.setCurrentPCAICAShowInstruction(Instructions_Annotations_PCA_ICA.Show_Instruction)
        else:
            self.show_annotations["pca_ica"] = True
            self.loaded_annotations["pca_ica"] = True
            self.m_scene.load_annotations("pca_ica")

    @QtCore.pyqtSlot()
    def show_hide_cnmfe_annotations(self):
        if self.loaded_annotations["cnmfe"]:

            if self.show_annotations["cnmfe"]:
                self.show_annotations["cnmfe"] = False
                self.m_scene.setCurrentCNMFEShowInstruction(Instructions_Annotations_CNMFE.Hide_Instruction)
            else:
                self.show_annotations["cnmfe"] = True
                self.m_scene.setCurrentCNMFEShowInstruction(Instructions_Annotations_CNMFE.Show_Instruction)
        else:
            self.show_annotations["cnmfe"] = True
            self.loaded_annotations["cnmfe"] = True
            self.m_scene.load_annotations("cnmfe")

    @QtCore.pyqtSlot()
    def exportMatfile(self):
        self.m_scene.export2Matfile()



if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = AnnotationWindow()
    w.resize(window_size[0], window_size[1])
    w.show()
    sys.exit(app.exec_())