import json
import redis
from deepface import DeepFace
from deepface.commons import functions


class Core:
    def __init__(self, org_id):
        self.redis = redis.StrictRedis(host="redis-19297.c300.eu-central-1-1.ec2.cloud.redislabs.com", port=19297, password="ddamvT9zi7xFplvRwMROdsBmRUbbfpJk")
        self.org_id = org_id

    def prepropcess(self, img):
        model = DeepFace.build_model("ArcFace")

        # Preprocessing face (detect face, resize, align, normalize)
        face = functions.preprocess_face(img, target_size=(112, 112), detector_backend='mtcnn')
        
        # Return predict representation (Representation = 512 vector)
        return model.predict(face)[0].tolist()
        

    def rset(self, person_id, img):
        # Preprocessing face (detect face, resize, align, normalize)
        img_preprocess = self.prepropcess(img)
        # Push representation to redis
        self.redis.rpush("{}:{}".format(self.org_id, person_id), json.dumps(img_preprocess))