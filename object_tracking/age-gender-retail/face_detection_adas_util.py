import sys
import time

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../../util')
from nms_utils import nms_boxes  # noqa
from detector_utils import letterbox_convert, reverse_letterbox  # noqa: E402

# ======================
# Parameters
# ======================

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 672

THRESHOLD = 0.5
IOU = 0.5


# ======================
# Secondaty Functions
# ======================

def convert_to_detector_object(bboxes, scores, im_w, im_h):
    detector_object = []
    for i in range(len(bboxes)):
        (x1, y1, x2, y2) = bboxes[i]
        score = scores[i]

        r = ailia.DetectorObject(
            category="",
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detector_object.append(r)

    return detector_object


# ======================
# Main functions
# ======================

def preprocess(img):
    #img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = letterbox_convert(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def decode_bbox(mbox_loc, mbox_priorbox, variances):
    mbox_loc = mbox_loc.reshape(-1, 4)
    mbox_priorbox = mbox_priorbox.reshape(-1, 4)
    variances = variances.reshape(-1, 4)

    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
    decode_bbox_center_y += prior_center_y
    decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
    decode_bbox_height *= prior_height

    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    bboxes = np.concatenate((
        decode_bbox_xmin[:, None],
        decode_bbox_ymin[:, None],
        decode_bbox_xmax[:, None],
        decode_bbox_ymax[:, None]), axis=-1)

    # bboxes = np.minimum(np.maximum(bboxes, 0.0), 1.0)

    return bboxes


def compute_face_detection_adas(model_info, img):
    score_th = THRESHOLD
    nms_th = IOU

    net = model_info['net']
    prior_box = model_info['prior_box']

    preprocess_img = preprocess(img)

    # feedforward
    output = net.predict([preprocess_img])
    mbox_loc, mbox_conf = output

    bboxes = decode_bbox(mbox_loc[0], prior_box[0], prior_box[1])

    mbox_conf = mbox_conf[0].reshape(-1, 2)
    cls_idx = 1
    i = mbox_conf[:, cls_idx] >= score_th
    bboxes = bboxes[i]
    scores = mbox_conf[i][:, 1]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * IMAGE_WIDTH
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * IMAGE_HEIGHT

    i = nms_boxes(bboxes, scores, nms_th)
    bboxes = bboxes[i].astype(int)
    scores = scores[i]

    detect_object = []
    for i in range(len(bboxes)):
        (x1, y1, x2, y2) = bboxes[i]
        w, h = (x2 - x1), (y2 - y1)
        cx, cy = x1 + w / 2, y1 + h / 2
        score = scores[i]

        r = ailia.DetectorObject(
            category=0,
            prob=score,
            x=(cx - w / 2) / IMAGE_WIDTH,
            y=(cy - h / 2) / IMAGE_HEIGHT,
            w=w / IMAGE_WIDTH,
            h=h / IMAGE_HEIGHT,
        )
        detect_object.append(r)

    detect_object = reverse_letterbox(detect_object, img, (IMAGE_HEIGHT, IMAGE_WIDTH))

    return detect_object

