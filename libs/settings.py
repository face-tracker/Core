class Setting:
    def __init__(self):
        self.enable_new_identity = True
        self.detection_minutes = 10

        # cosine distance < cosine_distance => found identity
        # cosine distance > cosine_distance and cosine distance < cosine_distance_new_identity => not found
        # cosine distance > cosine_distance_new_identity => new identity
        self.cosine_distance = 0.45
        self.cosine_distance_new_identity = 0.70

        self.min_face_size = 60
        self.fps = 30

        self.display_window = True

        self.image_limitation = 100

        self.face_db = "imgs"
