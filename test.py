from deepface import DeepFace
from deepface.commons import functions
import numpy as np
from libs.core import Core
import matplotlib.pyplot as plt
core = Core()

model = DeepFace.build_model("ArcFace")

core.reset_db()

for i in range(1, 4):
    print(i)
    # face = DeepFace.detectFace(img_path="{}.jpg".format(i), target_size=(112, 112), detector_backend='mtcnn')
    face = functions.preprocess_face("{}.jpg".format(i), target_size=(112, 112), detector_backend='mtcnn')
    # face = functions.normalize_input(face, normalization='ArcFace')
    # plt.imsave('t.jpg', face[0])
    reps = model.predict(face)[0].tolist()
    # local = DeepFace.represent(img_path=face[0], model_name='ArcFace',
                            #    model=model, detector_backend='mtcnn', normalization='ArcFace')
    core.represent_set(1, reps)
    # exit()

# represents = core.representations
# for i in represents:
#     re = np.array(i[1]).astype('float')
#     print(re.shape)
#     # DeepFace.detectFace(img_path=re, target_size=(112, 112), detector_backend='mtcnn')
#     # plt.imshow(re, interpolation='nearest')
#     exit()
