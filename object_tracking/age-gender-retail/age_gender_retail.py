import sys
import time

import ailia
import cv2
import numpy as np

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import get_base_parser, get_savepath, update_parser  # noqa: E402

sys.path.append('../blazeface')
from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'age-gender-recognition-retail-0013.onnx'
MODEL_PATH = 'age-gender-recognition-retail-0013.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/'

FACE_WEIGHT_PATH = 'blazefaceback.onnx'
FACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
FACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_IMAGE_HEIGHT = 256
FACE_IMAGE_WIDTH = 256
FACE_MIN_SCORE_THRESH = 0.5

IMAGE_SIZE = 62

# ======================
# Main functions
# ======================

def recognize_age_gender_retail(age_gender, frame):
    # frame is bgr
    net = age_gender["net"]
    detector = age_gender["detector"]

    # detect face
    detections = compute_blazeface(
        detector,
        frame,
        anchor_path='../age-gender-retail/anchorsback.npy',
        back=True,
        min_score_thresh=FACE_MIN_SCORE_THRESH
    )

    # adjust face rectangle
    new_detections = []
    for detection in detections:
        margin = 1.5
        r = ailia.DetectorObject(
            category=detection.category,
            prob=detection.prob,
            x=detection.x-detection.w*(margin-1.0)/2,
            y=detection.y-detection.h*(margin-1.0)/2-detection.h*margin/8,
            w=detection.w*margin,
            h=detection.h*margin,
        )
        new_detections.append(r)
    detections = new_detections

    # estimate age and gender
    for obj in detections:
        # get detected face
        margin = 1.0
        crop_img, top_left, bottom_right = crop_blazeface(
            obj, margin, frame
        )
        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
            continue

        img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
        img = np.expand_dims(img, axis=0)  # 次元合せ

        # inference
        output = net.predict([img])
        prob, age_conv3 = output
        prob = prob[0][0][0]
        age_conv3 = age_conv3[0][0][0][0]

        i = np.argmax(prob)
        gender = 'Female' if i == 0 else 'Male'
        age = round(age_conv3 * 100)

        return gender, age
    
    return None, None


def create_age_gender_retail(env_id):
    # model files check and download
    logger.info('=== age-gender-recognition model ===')
    check_and_download_models(
        WEIGHT_PATH, MODEL_PATH, REMOTE_PATH
    )
    logger.info('=== face detection model ===')
    check_and_download_models(
        FACE_WEIGHT_PATH, FACE_MODEL_PATH, FACE_REMOTE_PATH
    )

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id)
    return {"net": net, "detector": detector}


if __name__ == '__main__':
    main()
