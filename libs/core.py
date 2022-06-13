import json
import numpy as np
import redis
from .settings import Setting
from deepface.commons import distance



class Core:
    def __init__(self):
        self.settings = Setting()
        self.redis = redis.StrictRedis(host=self.settings.redis_host, port=self.settings.redis_port, password=self.settings.redis_password)
        self.representations = self.represent_get_all()

    def reset_db(self):
        self.redis.flushdb()

    def represent_set(self, person_id, representation):
        self.redis.rpush(str(person_id), json.dumps(representation))

    def represent_get_all(self):
        all_reps = []
        getPeople = self.redis.keys('*')
        for person in getPeople:
            reps = self.redis.lrange(person, 0, -1)
            for repJson in reps:
                # Set person id, representation
                r = np.array(json.loads(repJson)).astype('float')
                all_reps.append([person, r])
        return all_reps


    def find(self, target_representation):
        # Get difference between representations
        differences = []
        for representation in self.representations:
            diff = distance.findCosineDistance(representation[1], target_representation)
            differences.append([representation[0], diff])
        # Sort by difference ascending
        differences.sort(key=lambda x:x[1],reverse=False)

        if differences[0][1] < self.settings.cosine_distance:
            return False
        else:
            return differences[0]