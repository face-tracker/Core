# from deepface import DeepFace
# from deepface.commons import functions

# import matplotlib.pyplot as plt

import redis
redis = redis.StrictRedis(host="46.101.221.139", port=6379, password="123456")
print(redis.execute_command("ping"))
# redis.rpush("test", "test value")

# model = DeepFace.build_model("ArcFace")

# input_shape = (160, 160)

# source_path = "source.jpg"

# source = functions.preprocess_face(source_path, target_size=input_shape)

# print(source.shape)

# plt.imshow(source[0])

