
import os
from deepface import DeepFace
from deepface.basemodels import ArcFace
from cprint import *
import time
import imagesize
from pathlib import Path
import pickle
from os import path
import json

from numpy import identity


class Recognition:
    def __init__(self, cameras_path, db_path):
        self.cameras_path = cameras_path
        self.db_path = db_path
        self.model_name = 'ArcFace'
        self.model = ArcFace.loadModel()
        self.detector_backend = 'mtcnn'

    
    # Helpers Functions
    ## Create Directory if not exists
    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    # New Identity if not exists
    def new_identity(self, camera, image_path, image):
        cprint.ok("[{}] ## Creating new identity ##".format(camera))
        identity_name = str(int(time.time()))
        dir = os.path.join(self.db_path, identity_name)
        self.create_directory(dir)

        cprint.ok("[{}] ## New identity ` {} ` created successfully ##".format(camera, identity_name))

        new_directory_for_identity_path = os.path.join(dir, "{}_1.jpg".format(identity_name))
        # Moving image to new identity directory
        os.rename(image_path, new_directory_for_identity_path)
        time.sleep(0.5)
        # Learn new identity
        self.learn_new(camera, new_directory_for_identity_path)


    # Current identity
    def current_identity(self, camera, obj, image_path, image):
        # if (obj.ArcFace_cosine > 0.55):
        #     cprint.err("[{}] ## Distance is more than 55% ##".format(camera))
        #     self.new_identity(camera, image_path, image)
        # else:
        cprint.ok("[{}] ## Learning new angle for current identity ... ##".format(camera))
        if os.path.isfile(image_path) and os.path.isfile(obj.identity):
            # Getting identity name
            identity_name = os.path.dirname(Path(obj.identity)).split('\\')[-1]
            # Getting identity images count
            images_count = len(os.listdir(os.path.dirname(Path(obj.identity))))


            dir = os.path.join(os.path.dirname(Path(obj.identity)), "{}_{}.jpg".format(identity_name, images_count+1))
            os.rename(image_path,  dir)

            # Getting current identity analyze
            if (not os.path.exists(os.path.dirname(Path(obj.identity)) + '\\analyze.txt')):
                # Creating new analyze file
                f = open(os.path.dirname(Path(obj.identity)) + '\\analyze.json', 'w')
                analyze = DeepFace.analyze(dir, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                f.write(str(analyze))
                f.close()

        else:
            cprint.err("[{}] ## Learning success, but saving to database failed ##".format(camera))
        

    # Learn new identity
    def learn_new(self, camera, image_path):
        cprint.ok("[{}] ## Learning new idetity ##".format(camera))

        db_path = os.path.join(self.db_path, "representations_arcface.pkl")

        if path.exists(db_path):
            f = open(db_path, 'rb')
            # Loading current representations file
            representations = pickle.load(f)
            f.close()

            # Generating representation for new image
            embedding = DeepFace.represent( img_path = image_path,
                                            model = self.model,
                                            model_name = self.model_name,
                                            enforce_detection=False )
                                            
            # Adding new representation to current representations
            representations.append([
                image_path,
                embedding
            ])

            os.remove(db_path)
            time.sleep(1)
            # Saving new representations file
            f = open(db_path, "wb")
            pickle.dump(representations, f)
            time.sleep(1)

    # Directory has no directories
    def directory_has_no_directories(self, directory):
        return not os.listdir(directory)

    def start(self):
        # If there are cameras folders
        if self.directory_has_no_directories(self.cameras_path):
            cprint.err("No cameras found.")

        # If there are no cameras folders
        else:
            cprint.ok("### There are `{}` cameras found ###".format(len(os.listdir(self.cameras_path))))
            for camera in os.listdir(self.cameras_path):
                # Starting Processing
                cprint.warn('### Processing images from `{}` ###'.format(camera))

                # Get images from camera path
                images = os.listdir(os.path.join(self.cameras_path, camera))

                for image in images:
                    # Image path
                    image_path = os.path.join(self.cameras_path, camera, image)

                    # Get image size
                    width, height = imagesize.get(image_path)

                    # Check if image is too small
                    if (width < 24 and height < 24):
                        cprint.err("[{}] ## Image `{}` too small. Skipping... ##".format(camera, image))
                        os.remove(image_path)

                    # If image is is not small
                    else:
                        # Start image processing
                        cprint.ok('[{}] ## Processing `{}` image ##'.format(camera, image))

                        cprint.ok('[{}] ## Searching for the face in database ... ##'.format(camera))
                        try:
                            # Search for image in database
                            df = DeepFace.find( img_path = image_path,
                                                db_path = self.db_path,
                                                model_name = self.model_name,
                                                model = self.model,
                                                detector_backend = self.detector_backend,
                                                align = True,
                                                enforce_detection=True )
                        except:
                            cprint.err('[{}] ## Image search could not determine clear face ##'.format(camera))
                            pass
                        # if image has matches
                        if df.shape[0] > 0:
                            cprint.ok('[{}] ## Image has matches ##'.format(camera))

                            # Get top match
                            matched = df.iloc[0]

                            cprint.ok('[{}] ## Verifying face match with database ##'.format(camera))
                            try:
                                # Verify if match is valid
                                result = DeepFace.verify(   img1_path = image_path, 
                                                            img2_path = matched.identity,
                                                            model_name = self.model_name,
                                                            model = self.model,
                                                            detector_backend = self.detector_backend,
                                                            enforce_detection=True )
                            except:
                                cprint.err('[{}] ## Image verfication could not determine clear face ##'.format(camera))
                                pass

                            # If match is valid
                            if result['verified'] == True:
                                identity_name = matched.identity.split('\\')[1]
                                cprint.info('[{}] ## Verified with `{}` with cosine distance {} ## '.format(camera, identity_name, matched.ArcFace_cosine))
                                self.current_identity(camera, matched, image_path, image)
                            
                            # If match is not valid
                            else:
                                cprint.err('[{}] ## Verification failed ##'.format(camera))
                                # self.new_identity(camera, image_path, image)

                        # if image has no matches
                        else:
                            cprint.err('[{}] ## Image has no matches ##'.format(camera))
                            self.new_identity(camera, image_path, image)
                            # cprint.err('[{}] ## Face not detected ##'.format(camera))
