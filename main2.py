import json
import os
from os import path
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpngw import write_png
from PIL import Image

from deepface import DeepFace
from deepface.basemodels import ArcFace

model = ArcFace.loadModel()
deepface = DeepFace



# db_path = "imgs/representations_arcface.pkl"

# if path.exists(db_path):
#     f = open(db_path, 'rb')
#     # Loading current representations file
#     representations = pickle.load(f)
#     print(representations[0])
#     with open('data.txt', 'w') as my_data_file:
#         my_data_file.write(json.dumps(representations[0]))

# get images inside directory
def get_images(directory):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                images.append(os.path.join(root, file))
    return images


images = get_images("to_train")

for img in images:
    # face = mtcnn.detect(frame)
    # f = cv2.imread(face)
    # plt.imshow(image)

    # plt.imshow(f)
    # write_png('example1.png', face[:, :, ::-1].astype('uint8'))
    # Image.fromarray(face).convert("RGB").save("art.png")
    print(img)
    cv2.imwrite("filename.jpg", img)

    # np.array(face[:, :, ::-1])


    # print(face)
    exit()



print(images)