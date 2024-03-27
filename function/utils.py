import numpy as np
import base64
import cv2
from io import BytesIO

def decode_base642numpy(base_str):

    image_bytes = base64.b64decode(base_str)
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    return image_np

def decode_base642numpy_gray(base_str):

    image_bytes = base64.b64decode(base_str)
    image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    return image_np


def encode_numpy2base64(image):

    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return image_base64


def save_image(str, save_path):

    image = base64.b64decode(str)
    with open(save_path, 'wb') as file:
        file.write(image)


