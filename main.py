##########################################################################################
####                                  Import Libraries                                ####
##########################################################################################
import enum
import json
from multiprocessing import Queue
import threading
from deepface import DeepFace                                     #DeepFace library      #
from retinaface import RetinaFace                                 #RetinaFace library    #
import matplotlib.pyplot as plt                                   #Matplotlib library    #
import time                                                       #Time library          #
import cv2                                                        #OpenCV library        #
from cprint import *                                              #Colorful print library#
from pygrabber.dshow_graph import FilterGraph                     #PyGrabber library     #
from munch import DefaultMunch
### DETECTION LIBRARIESS ###
import torch                                                      #Pytorch library       #
##########################################################################################
####                                  Custom Libraries                                ####
##########################################################################################
from libs.camera import Camera                              #Camera Component   #
from libs.settings import Setting                           #Setting Component     #
from libs.connection import Connection                      #Connection Lib        #
from libs.api import Api                                    #API Lib               #
from libs.core import Core
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
####                                 CONNECTIONS                                      ####
##########################################################################################
api = Api(Connection())





##########################################################################################
####                                   Pre Process                                    ####
##########################################################################################


#### CREATING CAMERAS ####
# class Camera:
#     def __init__(self, id, source, name):
#         self.id = id
#         self.source = source
#         self.name = name



# cameras = [
#     Creating a camera object with the given source and name.
#     Camera('http://stream.shabakaty.com:6001/movies/ch14/ch14_720.m3u8', 'Camera 1'),
#     Camera('http://stream.shabakaty.com:6001/movies/ch2/ch2_720.m3u8', 'Camera 2'),
#     Camera('http://stream.shabakaty.com:6001/kids/ch6/ch6_720.m3u8', 'Camera 3'),
#     Camera('http://stream.shabakaty.com:6001/movies/ch3/ch3_360.m3u8', 'Camera 4'),
#     Camera('http://stream.shabakaty.com:6001/movies/ch8/ch8_720.m3u8', 'Camera 8'),
#     Camera('0', 'http://stream.shabakaty.com:6001/movies/ch5/ch5_720.m3u8', 'Fox Movies'),
#     Camera("https://cndw3.shabakaty.com/mp420-1080/56266C5D-3B0E-DEF0-F02C-62B26E9B57D1_video.mp4?response-content-disposition=attachment%3B%20filename%3D%22video.mp4%22&AWSAccessKeyId=RNA4592845GSJIHHTO9T&Expires=1651699211&Signature=ZKZRl20iDzzW%2Bvl8QcURD3ZLz2E%3D", "Before Sunrise")
#     Camera(1, 'DroidCam'),
#     Camera(0, 'HD Camera')
# ]
###########################################################################################




##########################################################################################
####                                    Core Process                                  ####
##########################################################################################

def camera_started(cameras_processes, camera):
    for process in cameras_processes:
        if process.name == camera.id:
            return process
    return None

def camera_stopped(cameras, camera_id):
    for camera in cameras:
        if str(camera['id']) == camera_id:
            return False
    return True

if __name__ == '__main__':
    #### Listing Cameras ####
    graph = FilterGraph()
    for name, index in enumerate(graph.get_input_devices()):
        cprint.warn("Camera name: {}, Camera index: {}".format(index, name))

    
    settings = Setting()
    starttime = time.time()
    cameras_processes = []

    while True:
        cprint.info("Checking Cameras...")
        # API GET all cameras
        all_cameras = api.get_all_cameras()
        # Start cameras
        for camera in all_cameras:
            camera = DefaultMunch.fromDict(camera)
            get_camera = camera_started(cameras_processes, camera)
            # Camera is active
            if get_camera is None:
                new_camera = Camera(camera = camera, device = device, settings = settings)
                new_camera.name = str(camera.id)
                new_camera.start()
                cameras_processes.append(new_camera)
                cprint.ok("Place [{}], Camera [{}] has been started".format(camera.place.id, camera.name))


        # Killing stopped cameras
        for camera in cameras_processes:
            if camera_stopped(all_cameras, camera.name):
                camera.kill()
                cameras_processes.remove(camera)
                cprint.warn("Camera [{}] has been stopped".format(camera.name))


        cprint.ok('Running on {} cameras'.format(len(cameras_processes)))

        time.sleep(30.0 - ((time.time() - starttime) % 30.0))

            
        
