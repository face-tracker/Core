import json
import numpy as np
import redis
from .settings import Setting
from deepface import DeepFace
from deepface.commons import distance, functions


class Core:
    def __init__(self, org_id):
        self.settings = Setting()
        self.redis = redis.StrictRedis(host=self.settings.redis_host, port=self.settings.redis_port, password=self.settings.redis_password)
        self.org_id = org_id
        self.representations = self.rget()

    # Reset redis database
    def reset_db(self):
        self.redis.flushdb()

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


    # Get all representations by organization id from redis
    def rget(self):
        all_reps = []

        # Getting all keys for an organization
        getPeople = self.redis.keys('{}:*'.format(self.org_id))

        # Looping for each person path (organization_id:person_id)
        for person in getPeople:
            # Get all representations for a person
            reps = self.redis.lrange(person, 0, -1)
            # Looping for each representation
            for repJson in reps:
                # Parse representation to numpy array and set values to float
                r = np.array(json.loads(repJson)).astype('float')
                # Set person id, representation
                all_reps.append([person.decode(encoding="utf-8").split(':')[1], r])
        return all_reps



    # Search for a person by representation similiarity
    def find(self, img):
        # Preprocessing face (detect face, resize, align, normalize)
        img_preprocess = self.prepropcess(img)

        # Get difference between representations
        differences = []
        for representation in self.representations:
            # diff = distance.findCosineDistance(representation[1], img_preprocess)
            diff = distance.findEuclideanDistance(distance.l2_normalize(representation[1]), distance.l2_normalize(img_preprocess))
            differences.append([representation[0], np.float64(diff)])
        # Sort by difference ascending
        differences.sort(key=lambda x:x[1],reverse=False)

        if differences[0][1] > self.settings.eeuclidean_l2_distance:
            return False
        else:
            return differences[0]


    def get_keys(self):
        return self.redis.keys('*')