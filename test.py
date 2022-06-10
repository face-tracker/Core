import json
import pickle
import cv2
from deepface import DeepFace
from deepface.commons import functions, distance
from PIL import Image
from matplotlib.font_manager import json_dump
import numpy as np

model = DeepFace.build_model("Facenet512")


from libs.api import Api
from libs.connection import Connection


api = Api(Connection())


local_representation = DeepFace.represent(img_path="source15.jpg", model_name='Facenet512',
                        model=model, detector_backend='mtcnn', normalization='Facenet')


# api.upload_represent({
#     "name": "embeddings_hassan",
#     "value": local_representation
# })

api_representation = api.download_represent("embeddings_hassan")
api_representation = np.array(api_representation['value']).astype('float')


diff = distance.findEuclideanDistance(distance.l2_normalize(local_representation), distance.l2_normalize(api_representation))
# DeepFace.find(img_path=[local_representation, api_representation], model_name='Facenet512', )

print(diff)

# image = np.array(face[:, :, ::-1])
# image = Image.fromarray((image * 1).astype(np.uint8))
# image = np.asarray(image)
# source = functions.preprocess_face(source_path="source.jpg", target_size=input_shape)


# cv2.imwrite("test.jpg", (face * 255)[:, :, ::-1])
