##########################################################################################
####                                  Import Libraries                                ####
##########################################################################################
import enum
from multiprocessing import Queue
import threading
from deepface import DeepFace                                     #DeepFace library      #
from retinaface import RetinaFace                                 #RetinaFace library    #
import matplotlib.pyplot as plt                                   #Matplotlib library    #
import time                                                       #Time library          #
import cv2                                                        #OpenCV library        #
from cprint import *                                              #Colorful print library#
from pygrabber.dshow_graph import FilterGraph                     #PyGrabber library     #

### DETECTION LIBRARIESS ###
import torch                                                      #Pytorch library       #
##########################################################################################
####                                  Custom Libraries                                ####
##########################################################################################
from python.final.libs.detection import Detection                        #Detection Component   #
from python.final.libs.recognition import Recognition                    #Recognition Component #
from python.final.libs.settings import Setting                           #Setting Component     #
##########################################################################################



##########################################################################################
####                                  For GPU support                                 ####
##########################################################################################
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus, 'GPU')
# for gpu in gpus:
#     print(gpu)
#     tf.config.experimental.set_memory_growth(gpu, True)

# 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cprint.ok('Running on device: {}'.format(device))

###########################################################################################



##########################################################################################
####                                   Pre Process                                    ####
##########################################################################################


#### CREATING CAMERAS ####
class Camera:
    def __init__(self, source, name):
        self.source = source
        self.name = name


cameras = [
    # Creating a camera object with the given source and name.
    # Camera('http://stream.shabakaty.com:6001/movies/ch14/ch14_720.m3u8', 'Camera 1'),
    # Camera('http://stream.shabakaty.com:6001/movies/ch2/ch2_720.m3u8', 'Camera 2'),
    # Camera('http://stream.shabakaty.com:6001/kids/ch6/ch6_720.m3u8', 'Camera 3'),
    # Camera('http://stream.shabakaty.com:6001/movies/ch3/ch3_360.m3u8', 'Camera 4'),
    # Camera('http://stream.shabakaty.com:6001/movies/ch8/ch8_720.m3u8', 'Camera 8'),
    Camera('http://stream.shabakaty.com:6001/movies/ch5/ch5_720.m3u8', 'Fox Movies'),
    # Camera("https://cndw3.shabakaty.com/mp420-1080/56266C5D-3B0E-DEF0-F02C-62B26E9B57D1_video.mp4?response-content-disposition=attachment%3B%20filename%3D%22video.mp4%22&AWSAccessKeyId=RNA4592845GSJIHHTO9T&Expires=1651699211&Signature=ZKZRl20iDzzW%2Bvl8QcURD3ZLz2E%3D", "Before Sunrise")
    # Camera(1, 'DroidCam'),
    # Camera(0, 'HD Camera')
]
###########################################################################################




##########################################################################################
####                                    Core Process                                  ####
##########################################################################################


if __name__ == '__main__':
    #### Listing Cameras ####
    graph = FilterGraph()
    for name, index in enumerate(graph.get_input_devices()):
        cprint.warn("Camera name: {}, Camera index: {}".format(index, name))


    repsQueue = Queue()
    settings = Setting()

    recognition = Recognition(repsQueue = repsQueue, db_path="imgs", settings = settings)

    recognition.start()

    for camera in cameras:
        # Creating a thread for each camera
        detection = Detection(camera = camera, repsQueue = repsQueue, device = device, settings = settings)
        detection.start()
