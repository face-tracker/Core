from deepface import DeepFace
from deepface.commons import functions
import numpy as np
from libs.core import Core
import matplotlib.pyplot as plt



core = Core(org_id=2)

# core.redis.scan()
# print(core.represent_get_by_org_id(1)[0][0])



core.reset_db()

for i in range(1, 4):
    print(i)
    core.rset(1, "{}.jpg".format(i))
    # exit()

# represents = core.representations
# for i in represents:
#     re = np.array(i[1]).astype('float')
#     print(re.shape)
#     # DeepFace.detectFace(img_path=re, target_size=(112, 112), detector_backend='mtcnn')
#     # plt.imshow(re, interpolation='nearest')
#     exit()
