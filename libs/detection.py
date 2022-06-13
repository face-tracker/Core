##########################################################################################
####                                  Import Libraries                                ####
##########################################################################################
import base64
import random
import time
from deepface import DeepFace                                     # DeepFace API         #
from deepface.commons import functions
import os                                                         #OS library            #
import threading                                                  #Threading library     #
import cv2                                                        #OpenCV library        #
from cprint import *                                              #Colorful print library#
from facenet_pytorch import MTCNN                                 #Facenet library       #
import datetime                                                   #DateTime library      #
import numpy as np                                                #Numpy library         #
from PIL import Image
from keras.preprocessing import image
import threading
from datetime import datetime, timedelta
from ttictoc import tic,toc
from multiprocessing import Process, Queue
from vidgear.gears import CamGear                                 #VidGear library       #
from deepface.basemodels import ArcFace

##########################################################################################



##########################################################################################
####                                    Core Process                                  ####
##########################################################################################
class Detection(Process):
    # Initializing
    def __init__(self, camera, repsQueue, device, settings):
        super(Detection, self).__init__()

        self.repsQueue = repsQueue

        self.camera = camera
        
        self.device = device

        self.settings = settings

        self.options = {
            # "CAP_PROP_FRAME_WIDTH": 1280, # resolution 2048x1152 - 1920x1080 - 1280x720 - 640x480 - 320x240 - 160x120
            # "CAP_PROP_FRAME_HEIGHT": 720,
            "CAP_PROP_FPS": self.settings.fps, # framerate
            "CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*'MJPG'), # codec
            'THREADED_QUEUE_MODE': True,
            # "CAP_PROP_BUFFERSIZE": 3
        }

        self.clean_faces = 0

        



    # Helper functions
    def echo(self, message, name = None):
        if name is None:
            cprint.ok("[Detection] " + message)
        else:
            cprint.ok("[Detection][{}] ".format(name) + message)
            

    # Important functions
    ## Start cameras
    def run(self):
        gear = CamGear(source=self.camera.source, **self.options)


        self.echo(message="Starting...", name=self.camera.name)
        gear.start()
        self.echo(message="Started!", name=self.camera.name)
        
        while True:
            # Request from API (response students ids)
            has_lecture_at_this_time = True
            if has_lecture_at_this_time:
                # Process cameras
                self.process_camera(gear)
            
            cprint.warn("[{}] Recalling lecture statue...".format(self.camera.name))
            time.sleep(60*1)

        # Stop cameras
        self.stop_cameras()

    ## Stop cameras
    def stop_cameras(self, gear):
        self.echo(message="Stopping...", name=self.camera.name)
        gear.stop()
        self.echo(message="Stopped!", name=self.camera.name)
        cv2.destroyAllWindows()


    ## Processing faces
    def process_faces(self, mtcnn, model, frame, camera_name):
        faces, _ = mtcnn.detect(frame)
        crop_img = frame
        reps = []
        # Representations
        if faces is not None:
            self.echo("Detect {} faces".format(len(faces)), name=camera_name)

            for index, box in enumerate(faces):
                # getting face
                crop_img = frame[
                    int(box[1]): int(box[3]),
                    int(box[0]): int(box[2])
                ]

                # Get height and width of the face
                height, width = crop_img.shape[:2]

                # Check if face width and height is valid
                if height > self.settings.min_face_size and width > self.settings.min_face_size:
                    # face = DeepFace.detectFace(img_path = crop_img, 
                    #                             target_size = (224, 224), 
                    #                             detector_backend = 'mtcnn',
                    #                             )
                    # if face is not None:
                    # self.repsQueue.put([camera_name, np.array(face * 255)])


                    face = crop_img
                    # cv2.imwrite('{}.jpg'.format(random.randint(0, 10000)), face)
                    try:
                        image = functions.preprocess_face(face, target_size=(112, 112), detector_backend='mtcnn')
                        represent = model.predict(image)[0].tolist()
                    except Exception as er:
                        cprint.err('## Not clear face ##')
                        continue

                    reps.append([camera_name, represent])

                    cprint.info("Face: {} from camera {} accepted.".format(index+1, camera_name))
                    cprint.info("######## QUEUE SIZE {} ############".format(self.repsQueue.qsize()))

                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    
                    self.clean_faces+=1


                    

                    # else:
                    #     cprint.err("Face: {} from camera {} rejected because face not clear.".format(index+1, camera_name))
                
                # Face too small
                else:
                    cprint.err("Face: {} from camera {} rejected because face is too small.".format(index+1, camera_name))

        for rep in reps:
            self.repsQueue.put(rep)
        return crop_img

    ## Processing cameras
    def process_camera(self, gear):
        counter = 0
        end_time = datetime.now() + timedelta(minutes=self.settings.detection_minutes)
        mtcnn = MTCNN(keep_all=True, device=self.device, image_size=112, select_largest=False)
        model = ArcFace.loadModel()

        # waiting time for m3u8 urls
        wt = 1 / self.settings.fps

        while True:
            start_time = time.time()

            current_time = datetime.now()
            # Finish after 5 minutes
            if current_time >= end_time:
                cprint.ok("[{}] Detection finished.".format(self.camera.name))
                break
            # if self.repsQueue.qsize() > 60:
            #     cprint.warn("Waiting Queue size: {}".format(self.repsQueue.qsize()))
            #     time.sleep(1)
            # else:
            frame = gear.read()
            if frame is None:
                break
            
            if counter % (self.settings.fps/2) == 0:
                try:
                    t = threading.Thread(target=self.process_faces, args=[mtcnn, model, frame, self.camera.name])
                    t.start()
                except Exception as er:
                    cprint.err(er)
                except:
                    continue
                
            if self.settings.display_window == True:
                cv2.putText(frame, text="Captured {} real face captures".format(self.clean_faces), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, org=(0, 20))
                cv2.imshow("Output {} - {}".format(self.camera.id, self.camera.name), frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Sleeping time for m3u8 urls
            dt = time.time() - start_time
            if wt - dt > 0:
                time.sleep(wt - dt)

            counter += 1


        cv2.destroyAllWindows()






