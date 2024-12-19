import os

def construct_downloadble_uri(image_path):
    image_name = os.path.split(image_path)[1]
    BASE_URI = "http://127.0.0.1:5000"
    UPLOAD_URI_PATH = "/download/{img_name}"
    return BASE_URI+UPLOAD_URI_PATH.format(img_name=image_name)