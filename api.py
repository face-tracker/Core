import cv2
import numpy as np
from libs.core import Core
from libs.api import Api
import tensorflow as tf
import argparse
from flask import Flask, request
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


# ------------------------------
# Service API Interface

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

@app.route('/', methods=['POST'])
def index():
    req = request.form
    image = request.files['image']
    person_id = req['person_id']
    organization_id = req['organization_id']
    secret = "secret"

    if secret != req['secret']:
        return {"message": "Not Authorized"}, 400

    if image.filename != '' and person_id != '' and organization_id != '':
        core = Core(organization_id)
        # Training image
        npimg = np.fromfile(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        core.rset(person_id, image)

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