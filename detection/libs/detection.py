##########################################################################################
####                                  Import Libraries                                ####
##########################################################################################
from deepface import DeepFace                                     # DeepFace API         #
import os                                                         #OS library            #
import threading                                                  #Threading library     #
import cv2                                                        #OpenCV library        #
from cprint import *                                              #Colorful print library#
from facenet_pytorch import MTCNN                                 #Facenet library       #
import datetime                                                   #DateTime library      #
##########################################################################################



##########################################################################################
####                                    Core Process                                  ####
##########################################################################################
class Detection():
    # Initializing
    def __init__(self, cameras, device):
        self.cameras = cameras

        
        self.device = device

        self.mtcnn = MTCNN(keep_all=True, image_size=224, device=device, select_largest=False, min_face_size=50, thresholds=[0.7, 0.8, 0.8])

        # Start cameras
        self.start_cameras()

        # Process cameras
        self.process_cameras()

        # Stop cameras
        self.stop_cameras()




    # Helper functions
    def echo(self, message, name = None):
        if name is None:
            cprint.ok("[Detection] " + message)
        else:
            cprint.ok("[Detection][{}] ".format(name) + message)

    # If directory empty create one
    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
            

    # Important functions
    ## Start cameras
    def start_cameras(self):
        for camera in self.cameras:
            self.echo(message="Starting...", name=camera.name)
            camera.gear.start()
            self.echo(message="Started!", name=camera.name)

    ## Stop cameras
    def stop_cameras(self):
        for camera in self.cameras:
            self.echo(message="Stopping...", name=camera.name)
            camera.gear.stop()
            self.echo(message="Stopped!", name=camera.name)
        cv2.destroyAllWindows()

    ## Processing faces
    def process_faces(self, frame, faces, camera_name):
        crop_img = frame
        if faces is not None:
            self.echo("Detect {} faces".format(len(faces)), name=camera_name)

            # Creating directory if not exists
            dir = self.create_directory("captured/{}".format(camera_name) + "/")

            for index, box in enumerate(faces):
                image_file_name = datetime.datetime.now().timestamp()

                crop_img = frame[
                    int(box[1]): int(box[3]),
                    int(box[0]): int(box[2])
                ]

                # Get height and width of the face
                height, width = crop_img.shape[:2]

                # Check if face width and height is valid
                if height > 50 and width > 50:
                    face = DeepFace.detectFace(img_path = crop_img, 
                                                target_size = (224, 224), 
                                                detector_backend = 'mtcnn',
                                                )
                    if face is not None:
                        cv2.imwrite("{}{}_{}.jpg".format(dir, image_file_name, index), (face * 255)[:, :, ::-1])
                        cprint.info("Face: {} from camera {} accepted.".format(index+1, camera_name))
                    else:
                        cprint.err("Face: {} from camera {} rejected because face not clear.".format(index+1, camera_name))
                
                # Face too small
                else:
                    cprint.err("Face: {} from camera {} rejected because face is too small.".format(index+1, camera_name))
                    
                # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # print(crop_img)


    ## Processing cameras
    def process_cameras(self):
        counter = 0

        while True:
            for index, camera in enumerate(self.cameras):
                frame = camera.gear.read()

                if frame is None:
                    break

                if counter % 30 == 0:
                    try:
                        faces, _ = self.mtcnn.detect(frame)
                        self.process_faces(frame, faces, camera.name)
                    except:
                        continue
            counter += 1





