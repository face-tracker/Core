
class Api:
    def __init__(self, connection):
        self.con = connection
    
    def get_all_cameras(self):
        print(self.con.get("cameras").content)
        return self.con.get("cameras").json()