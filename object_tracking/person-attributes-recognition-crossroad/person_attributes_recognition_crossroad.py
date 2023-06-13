import cv2
import sys
import time
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/person-attributes-recognition-crossroad/"


# ======================
# Arguemnt Parser Config
# ======================

model = "0234" # 0230 is small model

WEIGHT_PATH = "person-attributes-recognition-crossroad-{}.onnx".format(model)
MODEL_PATH = "person-attributes-recognition-crossroad-{}.onnx.prototxt".format(model)

def crop_and_resize(raw_data, x, y, w, h):
    # keep aspect
    if w * 2 < h:
        nw = h // 2
        x = x + (w - nw) // 2
        w = nw
    else:
        nh = w * 2
        y = y + (h - nh) // 2
        h = nh

    # crop
    input_data = np.zeros((h, w, 3))

    # zero padding
    iw = raw_data.shape[1]
    ih = raw_data.shape[0]
    if x < 0:
        w = w + x
        x = 0
    if y < 0:
        h = h + y
        y = 0
    if x + w > iw:
        w = iw - x
    if y + h > ih:
        h = ih - y
    input_data[0:h, 0:w, :] = raw_data[y:y+h, x:x+w, :]

    # resize
    input_data = cv2.resize(input_data, (80, 160))
    return input_data, x, y, w, h

# ======================
# Main functions
# ======================

def recognize_person_attributes_recognition_crossroad(crossroad, raw_data, x, y, w, h):
    # frame is bgr
    net = crossroad["net"]
    # get person
    input_data, x, y, w, h = crop_and_resize(raw_data, x, y, w, h)

    # get attribute
    if model == '0230':
        pass
    else:
        input_data = input_data.transpose(2,0,1)

    result = net.run(input_data)

    if model == '0230':
        classes = result[0][0][0][0]
    else:
        classes = result[0][0][:8]

    labels = ['is_male','has_bag','has_backpack','has_hat','has_longsleeves','has_longpants','has_longhair']
    return labels, classes


def create_person_attributes_recognition_crossroad(env_id):
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    return {"net": net}
