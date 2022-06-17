import urllib.parse


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