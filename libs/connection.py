import requests

class Connection:

    def __init__(self):
        self.api_url = "https://api.gruceing.dev/api/"
        self.headers = {}

        request = requests.post("{}auth/login".format(self.api_url), data={"email": "api@gmail.com", "password": "123456"})

        if request.status_code == 200:
            self.headers = {"Authorization": 'Bearer ' + request.json()["access_token"]}
        
        self.api_url += "python/"

    def post(self, endpoint, data):
        return requests.post("{}{}".format(self.api_url, endpoint), json=data, headers=self.headers)

    def get(self, endpoint):
        return requests.get("{}{}".format(self.api_url, endpoint), headers=self.headers)