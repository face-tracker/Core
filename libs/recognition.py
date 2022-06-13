
import os
import random
from deepface import DeepFace
from deepface.basemodels import ArcFace
from deepface.commons import functions
from cprint import *
import time
import imagesize
from pathlib import Path
import pickle
from os import path
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from numpy import identity
import threading
from multiprocessing import Process, Queue

from libs.core import Core



class Recognition(Process):
    def __init__(self, repsQueue, db_path, settings, *args, **kwargs):
        super(Recognition, self).__init__(*args, **kwargs)
        self.repsQueue = repsQueue
        self.db_path = db_path
        self.model_name = 'ArcFace'
        self.detector_backend = 'mtcnn'
        self.settings = settings



    # Helpers Functions
    ## Create Directory if not exists
    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    # New Identity if not exists
    def new_identity(self, camera, represent, model):
        cprint.ok("[{}] ## Creating new identity ##".format(camera))
        cprint.err("[{}] ## Learning new identity currently unavailable ##".format(camera))
        # identity_name = str(int(time.time()))
        # dir = os.path.join(self.db_path, identity_name)
        # self.create_directory(dir)

        # cprint.ok("[{}] ## New identity ` {} ` created successfully ##".format(camera, identity_name))

        # new_directory_for_identity_path = os.path.join(dir, "{}_1.jpg".format(identity_name))
        # cv2.imwrite(new_directory_for_identity_path, image[:, :, ::-1])

        # Moving image to new identity directory
        # time.sleep(0.5)
        # Learn new identity
        # self.learn_new(camera, new_directory_for_identity_path, model)


    # Current identity
    def current_identity(self, camera, diff, represent, model):
        if diff[1] > self.settings.cosine_distance_new_identity:
            cprint.ok("[{}] ## New identity ##".format(camera))
            self.new_identity(camera, represent, model)
        elif diff[1] > self.settings.cosine_distance and diff[1] < self.settings.cosine_distance_new_identity:
            cprint.err("[{}] ## Distance is more than {}% => {}% ##".format(camera, self.settings.cosine_distance * 100, diff[1] * 100))
        # else:
            cprint.ok("[{}] ## Matched with person id {} ... ##".format(camera, diff[0]))
            # cprint.ok("[{}] ## Learning new angle for current identity ... ##".format(camera))
            # if os.path.isfile(obj.identity):
            #     # Getting identity name
            #     identity_name = os.path.dirname(Path(obj.identity)).split('\\')[-1]
            #     # Getting identity images count
            #     images_files = os.listdir(os.path.dirname(Path(obj.identity)))
            #     images_count = len(images_files)

            #     if images_count > self.settings.image_limitation:
            #         cprint.err("[{}] ## Identity ` {} ` has more than 10 images ##".format(camera, identity_name))
            #         return False
            #     else:
            #         dir = os.path.join(os.path.dirname(Path(obj.identity)), "{}_{}.jpg".format(identity_name, images_count+1))

            #     cv2.imwrite(dir, image[:, :, ::-1])

            #     self.learn_new(camera, dir, model)

            #     # Getting current identity analyze
            #     # if (not os.path.exists(os.path.dirname(Path(obj.identity)) + '\\analyze.txt')):
            #     #     # Creating new analyze file
            #     #     f = open(os.path.dirname(Path(obj.identity)) + '\\analyze.json', 'w')
            #     #     analyze = DeepFace.analyze(dir, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            #     #     f.write(str(analyze))
            #     #     f.close()
            # else:
            #     cprint.err("[{}] ## Learning success, but saving to database failed ##".format(camera))
        

    # Learn new identity
    def learn_new(self, camera, image_path, model):
        cprint.ok("[{}] ## Learning new idetity ##".format(camera))

        db_path = os.path.join(self.db_path, "representations_arcface.pkl")

        if path.exists(db_path):
            f = open(db_path, 'rb')
            # Loading current representations file
            representations = pickle.load(f)
            f.close()

            # Generating representation for new image
            embedding = DeepFace.represent( img_path = image_path,
                                            model = model,
                                            model_name = self.model_name,
                                            enforce_detection=False )
                                            
            # Adding new representation to current representations
            representations.append([
                image_path,
                embedding
            ])

            # Saving new representations file
            f = open(db_path, "wb")
            pickle.dump(representations, f)

    # Directory has no directories
    def directory_has_no_directories(self, directory):
        return not os.listdir(directory)

    def run(self):
        model = ArcFace.loadModel()
        deepface = DeepFace

        #### Getting Redis Representations ####
        core = Core()
        redis_reps = core.representations
        if len(redis_reps) > 0:
            cprint.info("Found {} representations in Redis".format(len(redis_reps)))
        else:
            cprint.error("No representations found in Redis")
            exit(1)
        
        while True:
            cprint.err("Waiting...")
            # Getting representation (face) from detection thread queue
            rep = self.repsQueue.get()

            if rep:
                camera = rep[0]
                # Starting Processing
                cprint.warn('### Processing images from `{}` ###'.format(camera))

                # Get images from camera path
                # image = Image.fromarray((rep[1] * 1).astype(np.uint8))
                # image = np.asarray(image)
                represent = rep[1]


                cprint.ok('[{}] ## Searching for the face in database ... ##'.format(camera))
                try:
                    # Search for image in database
                    # df = deepface.find( img_path = image,
                    #                     db_path = self.db_path,
                    #                     model_name = self.model_name,
                    #                     model = model,
                    #                     detector_backend = self.detector_backend,
                    #                     align = True,
                    #                     enforce_detection=True )
                    # represent = DeepFace.represent(img_path=image, model_name='ArcFace',
                    #     model=model, detector_backend='mtcnn', normalization='ArcFace', enforce_detection=False)
                    # represent = functions.preprocess_face(image, target_size=(112, 112), detector_backend='mtcnn', enforce_detection=False)
                    # plt.imsave('{}.jpg'.format(random.randint(0, 10000)), represent[0])
 
                    # Getting cosine distance
                    dist = core.find(represent)
                    cprint.warn(dist)

                except Exception as er:
                    cprint.err('[{}] ## Image search could not determine clear face ##'.format(camera))
                    cprint.err(er)
                    # cv2.imwrite(str(int(time.time())) + ".jpg", image)
                    continue


                # if image has matches
                if dist:
                    cprint.ok('[{}] ## Image has matches ##'.format(camera))

                    # cprint.ok('[{}] ## Verifying face match with database ##'.format(camera))
                    # try:
                    #     # Verify if match is valid
                    #     result = DeepFace.verify(   img1_path = image, 
                    #                                 img2_path = matched.identity,
                    #                                 model_name = self.model_name,
                    #                                 model = model,
                    #                                 detector_backend = self.detector_backend,
                    #                                 enforce_detection=True )
                    # except:
                    #     cprint.err('[{}] ## Image verfication could not determine clear face ##'.format(camera))
                    #     pass

                    # # If match is valid
                    # if result['verified'] == True:
                    #     identity_name = matched.identity.split('\\')[1]
                    #     cprint.info('[{}] ## Verified with `{}` with cosine distance {} ## '.format(camera, identity_name, matched.ArcFace_cosine))
                    self.current_identity(camera, dist, represent, model)
                    
                    # If match is not valid
                    # else:
                        # cprint.err('[{}] ## Verification failed ##'.format(camera))
                        # self.new_identity(camera, image_path)

                # if image has no matches
                else:
                    cprint.err('[{}] ## Image has no matches ##'.format(camera))
                    if self.settings.enable_new_identity == True:
                        self.new_identity(camera, represent, model)
                    # cprint.err('[{}] ## Face not detected ##'.format(camera))
