import cv2
import numpy as np

import sys
sys.path.append('../../util')
from math_utils import softmax

num_coords = 16
x_scale = 128.0
y_scale = 128.0
h_scale = 128.0
w_scale = 128.0
min_score_thresh = 0.75
min_suppression_threshold = 0.3
num_keypoints = 6

# mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
kp1 = 1  # Left eye
kp2 = 0  # Right eye
theta0 = 0
dscale = 1.5
dy = 0.

resolution = 192


def resize_image(img, out_size, keep_aspect_ratio=True, return_scale_padding=False):
    """
    Resizes the input image to the desired size, keeping the original aspect
    ratio or not.

    Parameters
    ----------
    img: NumPy array
        The image to resize.
    out_size: int or (int, int)  (height, width)
        Resizes the image to the desired size.
    keep_aspect_ratio: bool (default: True)
        If true, resizes while keeping the original aspect ratio. Adds zero-
        padding if necessary.
    return_scale_padding: bool (default: False)
        If true, returns the scale and padding for each dimensions.

    Returns
    -------
    resized: NumPy array
        Resized image.
    scale: NumPy array, optional
        Resized / original, (scale_height, scale_width).
    padding: NumPy array, optional
        Zero padding (top, bottom, left, right) added after resizing.
    """
    img_size = img.shape[:2]
    if isinstance(out_size, int):
        out_size = np.array([out_size, out_size], dtype=int)
    else: # Assuming sequence of len 2
        out_size = np.array(out_size, dtype=int)
    scale = img_size / out_size
    padding = np.zeros(4, dtype=int)

    if img_size[0] != img_size[1] and keep_aspect_ratio:
        scale_long_side = np.max(scale)
        size_new = (img_size / scale_long_side).astype(int)
        padding = out_size - size_new
        padding = np.stack((padding // 2, padding - padding // 2), axis=1).flatten()
        scale[:] = scale_long_side
        resized = cv2.resize(img, (size_new[1], size_new[0]))
        resized = cv2.copyMakeBorder(resized, *padding, cv2.BORDER_CONSTANT, 0)
    else:
        resized = cv2.resize(img, (out_size[1], out_size[0]))

    if return_scale_padding:
        return resized, scale, padding
    else:
        return resized

def denormalize_detections(detections, resized_size, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        resized_size: size of the resized image (i.e. input image)
        scale: scalar that was used to resize the image
        pad: padding in the x (left) and y (top) dimensions

    """
    detections[:, 0] = (detections[:, 0] * resized_size - pad[0]) * scale
    detections[:, 1] = (detections[:, 1] * resized_size - pad[1]) * scale
    detections[:, 2] = (detections[:, 2] * resized_size - pad[0]) * scale
    detections[:, 3] = (detections[:, 3] * resized_size - pad[1]) * scale

    detections[:, 4::2] = (detections[:, 4::2] * resized_size - pad[1]) * scale
    detections[:, 5::2] = (detections[:, 5::2] * resized_size - pad[0]) * scale
    return detections

def detection2roi(detection, detection2roi_method='box'):
    """ Convert detections from detector to an oriented bounding box.

    Adapted from:
    mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    The center and size of the box is calculated from the center
    of the detected box. Rotation is calculated from the vector
    between kp1 and kp2 relative to theta0. The box is scaled
    and shifted by dscale and dy.

    """
    if detection2roi_method == 'box':
        # compute box center and scale
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = (detection[:, 3] - detection[:, 1])  # assumes square boxes

    elif detection2roi_method == 'alignment':
        # compute box center and scale
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        xc = detection[:, 4+2*kp1]
        yc = detection[:, 4+2*kp1+1]
        x1 = detection[:, 4+2*kp2]
        y1 = detection[:, 4+2*kp2+1]
        scale = np.sqrt(((xc-x1)**2 + (yc-y1)**2)) * 2
    else:
        raise NotImplementedError(
            "detection2roi_method [%s] not supported" % detection2roi_method)

    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:, 4+2*kp1]
    y0 = detection[:, 4+2*kp1+1]
    x1 = detection[:, 4+2*kp2]
    y1 = detection[:, 4+2*kp2+1]
    theta = np.arctan2(y0-y1, x0-x1) - theta0
    return xc, yc, scale, theta

def extract_roi(frame, xc, yc, theta, scale):
    # take points on unit square and transform them according to the roi
    points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]]).reshape(1, 2, 4)
    points = points * scale.reshape(-1, 1, 1)/2
    theta = theta.reshape(-1, 1, 1)
    R = np.concatenate((
        np.concatenate((np.cos(theta), -np.sin(theta)), 2),
        np.concatenate((np.sin(theta), np.cos(theta)), 2),
    ), 1)
    center = np.concatenate((xc.reshape(-1, 1, 1), yc.reshape(-1, 1, 1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps
    # these points back to the output square
    res = resolution
    points1 = np.array([[0, 0, res-1], [0, res-1, 0]], dtype='float32').T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].T.astype('float32')
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res), borderValue=127.5)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)
    if imgs:
        imgs = np.moveaxis(np.stack(imgs), 3, 1).astype('float32') / 127.5 - 1.0
        affines = np.stack(affines)
    else:
        imgs = np.zeros((0, 3, res, res))
        affines = np.zeros((0, 2, 3))

    return imgs, affines, points

def head_pose_preprocess(img, detections, scale, padding):
    """
    Preprocesses the image and face detections for the head pose estimator.

    Parameters
    ----------
    img: NumPy array
        The image to format.
    detections: NumPy array
        Face detections.
    scale: NumPy array
        Scale used when preprocessing the image for the face detection.
        Resized / original, (scale_height, scale_width)
    padding: NumPy array
        Padding used when preprocessing the image for the face detection.
        Zero padding (top, bottom, left, right) added after resizing

    Returns
    -------
    input_hp: NumPy array
        Formatted image.
    centers: NumPy array
        Center(s) (x, y) of the cropped faces.
    theta: NumPy array
        rotation angle(s) in radians of the cropping bounding boxes.
    """
    # Only handles detections from the 1st image
    detections = denormalize_detections(detections[0], 128, scale[0], padding[[0, 2]])
    xc, yc, scale, theta = detection2roi(detections)
    rois, _, _ = extract_roi(img, xc, yc, theta, scale)

    tmp = (np.moveaxis(rois, 1, -1) + 1) / 2
    input_hp = np.empty((tmp.shape[0], 224, 224, 3), dtype=tmp.dtype)
    for i in range(len(tmp)):
        input_hp[i] = cv2.resize(tmp[i], (224, 224))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, 3))
    input_hp = (input_hp - mean) / std
    input_hp = np.moveaxis(input_hp, -1, 1)

    return input_hp, np.stack((xc, yc), axis=1), theta

def head_pose_postprocess(preds_hp, theta):
    """
    Postprocesses the raw head pose predictions (scores for yaw, pitch, roll)
    and returns the head poses (roll, yaw, pitch) in radians.

    Parameters
    ----------
    preds_hp: NumPy array
        Raw head pose predictions.
    theta: NumPy array
        rotation angle(s) in radians of the cropping bounding boxes.

    Returns
    -------
    head_pose: NumPy array
        Roll (left+), yaw (right+), pitch (down+) in radians in the original
        image coordinates.
    """
    head_pose = np.empty((len(preds_hp[0]),3), dtype=np.float32)
    for i_new, i in enumerate([2, 0, 1]):
        score = preds_hp[i]
        pred = softmax(score)
        tmp = (pred * np.arange(66)[np.newaxis]).sum(axis=1)
        head_pose[:, i_new] = (tmp * 3 - 99)
    # At this point, we have roll left+, yaw right+, pitch up+ in degrees
    head_pose *= np.pi / 180
    head_pose[:, 0] += theta
    head_pose[:, 2] *= -1 # pitch down+
    return head_pose
