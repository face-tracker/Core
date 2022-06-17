from deepface import DeepFace
from deepface.commons import functions
import numpy as np
from libs.core import Core
from libs.connection import Connection
from libs.api import Api
import matplotlib.pyplot as plt
from deepface.basemodels import ArcFace, Boosting, DeepID, DlibResNet, DlibWrapper, Facenet, Facenet512, FbDeepFace, OpenFace, VGGFace
from ttictoc import tic, toc


# models = [
#     'VGG-Face',
#     'OpenFace',
#     'Facenet',
#     'Facenet512',
#     # 'DeepFace',
#     # 'DeepID',
#     'Dlib',
#     'ArcFace',
#     # 'SFace',
#     # 'Emotion',
#     # 'Age',
#     # 'Gender',
#     # 'Race'
# ]
# checks = []
# for model in models:
#     print("==========================================================")
#     print("Model: {}".format(model))
#     tic()
# model = DeepFace.build_model("Facenet512")
# input_shape_x, input_shape_y = functions.find_input_shape(model)
# print(input_shape_x, input_shape_y)
#     face = functions.preprocess_face("1.jpg", target_size=(input_shape_x, input_shape_y), detector_backend='mtcnn')
#     model.predict(face)[0].tolist()
#     checks.append("{}:{}".format(model_name, toc()))

# for check in checks:
#     print(check)


# for detector in ['opencv', 'ssd', 'dlib','mtcnn', 'retinaface']:
#     tic()
#     face = functions.preprocess_face("1.jpg", detector_backend=detector)
#     checks.append("{}:{}".format(detector, toc()))


# for check in checks:
#     print(check)


core = Core(org_id=2, connection=Connection())

for i in range(1, 4):
    print(i)
    core.rset(4, "{}.jpg".format(i))

for i in range(4, 7):
    print(i)
    core.rset(5, "{}.jpg".format(i))

for i in range(7, 10):
    print(i)
    core.rset(6, "{}.jpg".format(i))

