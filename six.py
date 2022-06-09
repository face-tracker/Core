import requests

while True:
    try:
        request = requests.post("http://ps-control-center.com/res/getByAction_view.php",
                            data={"action": "getRes", "stdID_ajax": "22114160079"})
        if (request.status_code == 200):
            print(request.json())
            exit()
        else:
            print("Error: {}".format(request.status_code))
    except:
        pass
