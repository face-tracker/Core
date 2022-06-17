import json
import numpy as np
import redis
from .settings import Setting
from deepface import DeepFace
from deepface.commons import distance, functions


class Core:
    def __init__(self, org_id, connection):
        self.settings = Setting()
        # self.redis = redis.StrictRedis(host=self.settings.redis_host, port=self.settings.redis_port, password=self.settings.redis_password)
        self.org_id = org_id
        self.con = connection
        self.representations = self.rget()

    # Reset redis database
    # def reset_db(self):
    #     self.redis.flushdb()

    def prepropcess(self, img):
        model = DeepFace.build_model("ArcFace")

        # Preprocessing face (detect face, resize, align, normalize)
        face = functions.preprocess_face(
            img, target_size=(112, 112), detector_backend='mtcnn')

        # Return predict representation (Representation = 512 vector)
        return model.predict(face)[0].tolist()

    def rset(self, person_id, img):
        # Preprocessing face (detect face, resize, align, normalize)
        img_preprocess = self.prepropcess(img)
        # Push representation to redis

        upload = self.con.post("representations/add", {
            "person_id": person_id,
            "organization_id": self.org_id,
            "data": json.dumps(img_preprocess)
        }, [('image', (img, open(img, 'rb'), 'image/jpeg'))])
        print(upload.json())

        # self.redis.rpush("{}:{}".format(self.org_id, person_id),
        #                  json.dumps(img_preprocess))

    # Get all representations by organization id from redis

    def rget(self):
        all_reps = []
        getRepresentations = self.con.get("representations/{}".format(self.org_id)).json()
        # Looping for each person path (organization_id:person_id)
        for representation in getRepresentations:
            # Parse representation to numpy array and set values to float
            r = np.array(json.loads(representation['data'])).astype('float')
            r = json.loads(representation['data'])
            # Set person id, representation
            all_reps.append(
                [representation['person_id'], r])
        # print(all_reps)
        return all_reps

    # Search for a person by representation similiarity

    def find(self, img):
        # Preprocessing face (detect face, resize, align, normalize)
        if type(img) == list:
            img_preprocess = img
        else:
            img_preprocess = self.prepropcess(img)

        # Get difference between representations
        differences = []
        for representation in self.representations:
            # diff = distance.findCosineDistance(representation[1], img_preprocess)
            diff = distance.findEuclideanDistance(distance.l2_normalize(
                representation[1]), distance.l2_normalize(img_preprocess))
            differences.append([representation[0], np.float64(diff)])
        # Sort by difference ascending
        differences.sort(key=lambda x: x[1], reverse=False)

        if differences[0][1] > self.settings.eeuclidean_l2_distance:
            return False
        else:
            return differences[0]

    # def get_keys(self):
    #     return self.redis.keys('*')
