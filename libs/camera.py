##########################################################################################
####                                  Import Libraries                                ####
##########################################################################################
import time
import threading  # Threading library     #
import cv2  # OpenCV library        #
from cprint import *  # Colorful print library#
from facenet_pytorch import MTCNN  # Facenet library       #
import datetime  # DateTime library      #
import threading
from datetime import datetime, timedelta
from multiprocessing import Process
from vidgear.gears import CamGear  # VidGear library       #
from deepface.detectors import FaceDetector
from libs.api import Api
from libs.core import Core
from libs.connection import Connection
from memprof import *
from threading import active_count
from PIL import Image

##########################################################################################




##########################################################################################
####                                    Core Process                                  ####
##########################################################################################
class Camera(Process):
    # Initializing
    def __init__(self, camera, device, settings):
        super(Camera, self).__init__()

        self.camera = camera

        self.device = device

        self.settings = settings

        self.options = {
            # resolution 2048x1152 - 1920x1080 - 1280x720 - 640x480 - 320x240 - 160x120
            "CAP_PROP_FRAME_WIDTH": self.camera.width,
            "CAP_PROP_FRAME_HEIGHT": self.camera.height,
            "CAP_PROP_FPS": self.camera.fps,  # framerate
            "CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*'MJPG'),  # codec
            'THREADED_QUEUE_MODE': True,
            # "CAP_PROP_BUFFERSIZE": 3
        }

        self.clean_faces = 0

    # Helper functions

    def echo(self, message, name=None):
        if name is None:
            cprint.ok("[Detection] " + message)
        else:
            cprint.ok("[Detection][{}] ".format(name) + message)

    # Important functions
    # Start cameras
    def run(self):
        gear = CamGear(source=self.camera.source, **self.options)

        self.echo(message="Starting...", name=self.camera.name)
        gear.start()
        self.echo(message="Started!", name=self.camera.name)

        con = Connection()

        core = Core(self.camera.organization_id, connection=con)

        api = Api(con)


        if len(core.representations) == 0:
            self.echo(message="No representations found for organization {} - {}".format(
                self.camera.organization.id, self.camera.organization.name), name=self.camera.name)
            api.toggle_camera(self.camera.id)
            return

        mtcnn = MTCNN(keep_all=True, device=self.device, image_size=112, select_largest=False, min_face_size=self.camera.min_face_size)
        self.process_camera(gear, mtcnn, core, api)

        # Stop cameras
        self.stop_cameras()

    # Stop cameras
    def stop_cameras(self, gear):
        self.echo(message="Stopping...", name=self.camera.name)
        gear.stop()
        self.echo(message="Stopped!", name=self.camera.name)
        cv2.destroyAllWindows()

    # Processing faces
    def process_faces(self, mtcnn, core, api, frame, camera):
        faces, _ = mtcnn.detect(frame)
        crop_img = frame
        # Representations
        if faces is not None:
            self.echo("Detect {} faces".format(len(faces)), name=camera.name)

            for index, box in enumerate(faces):
                _frame = frame
                # getting face
                crop_img = frame[
                    int(box[1]): int(box[3]),
                    int(box[0]): int(box[2])
                ]

                # Get height and width of the face
                height, width = crop_img.shape[:2]

                # Check if face width and height is valid
                # if height > camera.min_face_size and width > camera.min_face_size:
                if True:
                    face = crop_img
                    try:
                        dist = core.find(face)
                        if dist:
                            cprint.info("[{}] Verified with person ( {} )".format(
                                camera.name, dist[0]))

                            # Text on frame
                            cv2.putText(frame, text="Captured {} - By {} at {}".format(
                            time.strftime("%I:%M%p on %B %d, %Y"), camera.name, camera.place.name), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, org=(0, 50))
                            
                            # Rectancgle on frame
                            cv2.rectangle(_frame, (int(box[0]), int(box[1])), (int(
                            box[2]), int(box[3])), (0, 255, 0), 2)

                            # Person id on frame
                            cv2.putText(frame, text="Person id: {}".format(dist[0]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1, org=(int(box[0]), int(
                            box[1]) - 10))

                            # Send to API
                            api.track_person(dist[0], camera.id, camera.organization.id, _frame)
                        else:
                            cprint.err(
                                "[{}] Face not found!".format(camera.name))

                        # cprint.info("Face: {} from camera {} accepted.".format(index+1, camera.name))

                        
                        
                        

                        self.clean_faces += 1

                    except Exception as er:
                        cprint.err('## Not clear face ##')
                        cprint.err(er)
                        continue

                # Face too small
                else:
                    cprint.err("Face: {} from camera {} rejected because face is too small.".format(
                        index+1, camera_name))

        return crop_img

    # Processing cameras
    def process_camera(self, gear, mtcnn, core, api):
        counter = 0

        end_time = datetime.now() + timedelta(minutes=self.camera.type)

        # waiting time for m3u8 urls
        wt = 1 / self.settings.fps

        while True:
            start_time = time.time()
            current_time = datetime.now()
            # Finish after x minutes or continuous mode
            if current_time >= end_time and not self.camera.type == 0:
                cprint.ok("[{}] Detection finished.".format(self.camera.name))
                break

            frame = gear.read()
            if frame is None:
                break


            if counter % self.settings.fps == 0:
                try:
                    t = threading.Thread(target=self.process_faces, args=[
                                         mtcnn, core, api, frame, self.camera])
                    t.start()
                except Exception as er:
                    cprint.err(er)
                    pass

            if self.settings.display_window == True:
                cv2.putText(frame, text="Captured {} real face captures".format(
                    self.clean_faces), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, org=(0, 20))
                cv2.imshow("Output {} - {} - {}".format(self.camera.organization.id, self.camera.id,
                           self.camera.name), frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # Sleeping time for m3u8 urls
            dt = time.time() - start_time
            if wt - dt > 0:
                time.sleep(wt - dt)

            counter += 1
            # cprint.info(active_count())

        cv2.destroyAllWindows()
