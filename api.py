import cv2
import numpy as np
from libs.core import Core
from libs.connection import Connection
from libs.api import Api
from deepface import DeepFace
model = DeepFace.build_model("ArcFace")
from deepface.commons import functions
import tensorflow as tf
import argparse
from flask import Flask, jsonify, request, make_response
import os
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------------------


tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

app = Flask(__name__)

if tf_version == 1:
	graph = tf.get_default_graph()


api  = Api(Connection())
core = Core()

# ------------------------------
# Service API Interface

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']

@app.route('/')
def index():
	return '<h1>Hello, world!</h1>'


@app.route('/train', methods=['POST'])
def train():
    req = request.form
    image = request.files['image']
    person_id = req['person_id']
    if image.filename != '' and person_id != '':
        # Training image
        npimg = np.fromfile(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        face = functions.preprocess_face(image, target_size=(112, 112), detector_backend='mtcnn')
        reps = model.predict(face)[0].tolist()
        core.represent_set(person_id, reps)

        ###################################
        ###### [ Send image to api ] ######
        ###################################
        

        return {"message": "Success"}, 200
    else:
        return {"message": "Error"}, 400





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--port',
		type=int,
		default=5000,
		help='Port of serving api')
	args = parser.parse_args()
	app.run(host='0.0.0.0', port=args.port)