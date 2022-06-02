import json
import os
from os import path
import pickle

db_path = "imgs/representations_arcface.pkl"

if path.exists(db_path):
    f = open(db_path, 'rb')
    # Loading current representations file
    representations = pickle.load(f)
    print(representations[0])
    with open('data.txt', 'w') as my_data_file:
        my_data_file.write(json.dumps(representations[0]))