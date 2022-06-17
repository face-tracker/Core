import requests
from .settings import Setting 
class Connection:

    def __init__(self):
        settings = Setting()
        self.api_url = "https://api.gruceing.dev/api/"
        self.headers = {}

        request = requests.post("{}auth/login".format(self.api_url), data={"email": settings.api_email, "password": settings.api_password})

        if request.status_code == 200:
            self.headers = {"Authorization": 'Bearer ' + request.json()["access_token"]}
        
        self.api_url += "python/"
    
    def post(self, endpoint, data, files=None):
        return requests.post("{}{}".format(self.api_url, endpoint), data=data, headers=self.headers, files=files)

    def get(self, endpoint):
        return requests.get("{}{}".format(self.api_url, endpoint), headers=self.headers)