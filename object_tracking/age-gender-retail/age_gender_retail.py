import sys
import time

import ailia
import cv2
import numpy as np
import math

sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import imread, load_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from blazeface_utils import compute_blazeface, crop_blazeface  # noqa: E402
from face_detection_adas_util import compute_face_detection_adas  # noqa
import hopenet_utils as hut

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'age-gender-recognition-retail-0013.onnx' # 5ms
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

#DETECTION_MODEL_TYPE = "blazeface" # 28ms
DETECTION_MODEL_TYPE = "face-detection-adas" # 53ms

if DETECTION_MODEL_TYPE == "blazeface":
    FACE_WEIGHT_PATH =BLAZEFACE_WEIGHT_PATH
    FACE_MODEL_PATH = BLAZEFACE_MODEL_PATH
    FACE_REMOTE_PATH = BLAZEFACE_REMOTE_PATH
else:
    FACE_WEIGHT_PATH =FACE_DETECTION_ADAS_WEIGHT_PATH
    FACE_MODEL_PATH = FACE_DETECTION_ADAS_MODEL_PATH
    FACE_REMOTE_PATH = FACE_DETECTION_ADAS_REMOTE_PATH

HEAD_POSE_MODEL_NAME = 'hopenet_lite' # 25ms
#HEAD_POSE_MODEL_NAME = 'hopenet_robust_alpha1' #59ms
HEAD_POSE_WEIGHT_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx'
HEAD_POSE_MODEL_PATH = f'{HEAD_POSE_MODEL_NAME}.onnx.prototxt'
HEAD_POSE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models/hopenet/'

FACE_IMAGE_HEIGHT = 256
FACE_IMAGE_WIDTH = 256
FACE_MIN_SCORE_THRESH = 0.5

HEAD_POSE_IMAGE_SIZE = 224

IMAGE_SIZE = 62

PROFILE = False

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


def head_pose_estimation(crop_img, hp_estimator):
    img = cv2.resize(crop_img, (HEAD_POSE_IMAGE_SIZE, HEAD_POSE_IMAGE_SIZE))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, 3))
    img = (img / 255.0 - mean) / std
    img = np.moveaxis(img, -1, 1)
    preds_hp = hp_estimator.run(img)
    theta = 0.0
    head_poses = hut.head_pose_postprocess(preds_hp, theta)
    return head_poses


def age_gender_estimation(crop_img, net):
    img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.expand_dims(img, axis=0)  # 次元合せ
    output = net.predict([img])
    prob, age_conv3 = output
    prob = prob[0][0][0]
    age_conv3 = age_conv3[0][0][0][0]

    i = np.argmax(prob)
    gender = 'Female' if i == 0 else 'Male'
    age = round(age_conv3 * 100)
    return age, gender


def draw_head_pose(crop_img, head_pose):
    p1 = (crop_img.shape[1] // 2, crop_img.shape[0] // 2)
    p2 = (int(p1[0] - p1[0] * math.sin(head_pose[1])), int(p1[1] + p1[1] * math.sin(head_pose[2])))
    l = max(crop_img.shape[1], crop_img.shape[0]) // 20
    if l < 1:
        l = 1
    cv2.line(crop_img, p1, p2, (255,0,0), thickness=l)


# ======================
# Main functions
# ======================

def recognize_age_gender_retail(age_gender, frame):
    # frame is bgr
    net = age_gender["net"]
    detector = age_gender["detector"]
    hp_estimator = age_gender["hp_estimator"]

    # detect face
    if PROFILE:
        start = int(round(time.time() * 1000))
    detections = detector(frame)
    if PROFILE:
        end = int(round(time.time() * 1000))
        logger.info(f'\tface detection processing time {end - start} ms')

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

        # Age gender estimation
        if PROFILE:
            start = int(round(time.time() * 1000))
        age, gender = age_gender_estimation(crop_img, net)
        if PROFILE:
            end = int(round(time.time() * 1000))
            logger.info(f'\tage_gender_estimation processing time {end - start} ms')

        # Head pose estimation
        if PROFILE:
            start = int(round(time.time() * 1000))
        head_poses = head_pose_estimation(crop_img, hp_estimator)
        if PROFILE:
            end = int(round(time.time() * 1000))
            logger.info(f'\thead_pose_estimation processing time {end - start} ms')

        draw_head_pose(crop_img, head_poses[0])
        if abs(head_poses[0][1]) >= math.pi/4 or abs(head_poses[0][2]) >= math.pi/4: # 45 degree
            return None, None, crop_img

        return gender, age, crop_img
    
    return None, None, frame


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
    logger.info('=== face direction model ===')
    check_and_download_models(
        HEAD_POSE_WEIGHT_PATH, HEAD_POSE_MODEL_PATH, HEAD_POSE_REMOTE_PATH
    )

    # net initialize
    net = ailia.Net(
        MODEL_PATH, WEIGHT_PATH, env_id=env_id
    )
    detector = ailia.Net(FACE_MODEL_PATH, FACE_WEIGHT_PATH, env_id=env_id)
    detector = setup_detector(detector)
    hp_estimator = ailia.Net(
            HEAD_POSE_MODEL_PATH, HEAD_POSE_WEIGHT_PATH, env_id=env_id
    )
    hp_estimator.set_input_shape((1, HEAD_POSE_IMAGE_SIZE, HEAD_POSE_IMAGE_SIZE, 3))

    return {"net": net, "detector": detector, "hp_estimator": hp_estimator}


if __name__ == '__main__':
    main()
