import os
import time
from cprint import *
from recognition.libs.recognition import Recognition

cameras_path = 'captured'
db_path = 'imgs'

recognition = Recognition(cameras_path=cameras_path, db_path=db_path)


cycle_per_minute = 1 * 60
while True:
    cpt = sum([len(files) for r, d, files in os.walk(cameras_path)])
    if (cpt > 0):
        # Starting recognition
        recognition.start()
    else:
        # No cameras found
        cprint.err("No cameras found. retry after {} seconds".format(cycle_per_minute))
    
    time.sleep(cycle_per_minute)















