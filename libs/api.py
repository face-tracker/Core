import io
import urllib.parse
from PIL import Image

class Api:
    def __init__(self, connection):
        self.con = connection
    
    def get_all_cameras(self):
        # print(self.con.get("cameras").json())
        return self.con.get("cameras").json()
    
    def toggle_camera(self, id):
        self.con.post("cameras/state", {"id": id})
    
    def upload_represent(self, rep):
        self.con.post("redis/set", rep)

    # def download_represent(self, rep):
    #     print(urllib.parse.quote("redis/get/" + rep))
    #     return self.con.get(urllib.parse.quote("redis/get/" + rep)).json()

    def track_person(self, person_id, camera_id, organization_id, image):
        im = Image.fromarray((image.astype("uint8"))[:, :, ::-1])

        upload_image = io.BytesIO()
        im.save(upload_image, "JPEG")
        upload_image.seek(0)
                
        self.con.post("trackings/new", {
            "person_id": person_id,
            "camera_id": camera_id,
            "organization_id": organization_id
        }, [('image', ("tracking", upload_image, 'image/jpeg'))])
        