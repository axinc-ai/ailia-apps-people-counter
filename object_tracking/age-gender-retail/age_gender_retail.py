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

from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402
from face_detection_adas_util import compute_face_detection_adas  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'age-gender-recognition-retail-0013.onnx'
MODEL_PATH = 'age-gender-recognition-retail-0013.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/age-gender-recognition-retail/'

BLAZEFACE_WEIGHT_PATH = 'blazefaceback.onnx'
BLAZEFACE_MODEL_PATH = 'blazefaceback.onnx.prototxt'
BLAZEFACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
BLAZEFACE_ANCHOR_PATH = '../age-gender-retail/anchorsback.npy'

FACE_DETECTION_ADAS_WEIGHT_PATH = 'face-detection-adas-0001.onnx'
FACE_DETECTION_ADAS_MODEL_PATH = 'face-detection-adas-0001.onnx.prototxt'
FACE_DETECTION_ADAS_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/face-detection-adas/'
FACE_DETECTION_ADAS_PRIORBOX_PATH = '../age-gender-retail/mbox_priorbox.npy'

DETECTION_MODEL_TYPE = "blazeface"
#DETECTION_MODEL_TYPE = "face-detection-adas"

if DETECTION_MODEL_TYPE == "blazeface":
    FACE_WEIGHT_PATH =BLAZEFACE_WEIGHT_PATH
    FACE_MODEL_PATH = BLAZEFACE_MODEL_PATH
    FACE_REMOTE_PATH = BLAZEFACE_REMOTE_PATH
else:
    FACE_WEIGHT_PATH =FACE_DETECTION_ADAS_WEIGHT_PATH
    FACE_MODEL_PATH = FACE_DETECTION_ADAS_MODEL_PATH
    FACE_REMOTE_PATH = FACE_DETECTION_ADAS_REMOTE_PATH

FACE_IMAGE_HEIGHT = 256
FACE_IMAGE_WIDTH = 256
FACE_MIN_SCORE_THRESH = 0.5

IMAGE_SIZE = 62


# ======================
# Secondaty Functions
# ======================

def setup_detector(net):
    if DETECTION_MODEL_TYPE == 'blazeface':
        from blazeface_utils import compute_blazeface  # noqa

        def _detector(img):
            detections = compute_blazeface(
                net,
                img,
                anchor_path=BLAZEFACE_ANCHOR_PATH,
                back=True,
                min_score_thresh=FACE_MIN_SCORE_THRESH
            )

            # adjust face rectangle
            detect_object = []
            for d in detections:
                margin = 1.5
                r = ailia.DetectorObject(
                    category=d.category,
                    prob=d.prob,
                    x=d.x - d.w * (margin - 1.0) / 2,
                    y=d.y - d.h * (margin - 1.0) / 2 - d.h * margin / 8,
                    w=d.w * margin,
                    h=d.h * margin,
                )
                detect_object.append(r)

            return detect_object

        detector = _detector
    else:
        prior_box = np.squeeze(np.load(FACE_DETECTION_ADAS_PRIORBOX_PATH))

        model_info = {
            'net': net,
            'prior_box': prior_box,
        }

        def _detector(img):
            im_h, im_w, _ = img.shape
            detections = compute_face_detection_adas(model_info, img)

            # adjust face rectangle
            detect_object = []
            for d in detections:
                enlarge = 1.2
                r = ailia.DetectorObject(
                    category=d.category,
                    prob=d.prob,
                    x=d.x - d.w * (enlarge - 1.0) / 2,
                    y=d.y - d.h * (enlarge - 1.0) / 2,
                    w=d.w * enlarge,
                    h=d.h * enlarge,
                )
                detect_object.append(r)

            return detect_object

        detector = _detector

    return detector


# ======================
# Main functions
# ======================

def recognize_age_gender_retail(age_gender, frame):
    # frame is bgr
    net = age_gender["net"]
    detector = age_gender["detector"]

    # detect face
    detections = detector(frame)

    # sort by prob
    max_obj = None
    for obj in detections:
        if max_obj == None or max_obj.prob < obj.prob:
            max_obj = obj

    # estimate age and gender
    if max_obj != None:
        detections = [max_obj]
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

        return gender, age, crop_img
    
    return None, None, None


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
    detector = setup_detector(detector)
    return {"net": net, "detector": detector}


if __name__ == '__main__':
    main()
