import sys
import time

import numpy as np
import cv2
from matplotlib import cm

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

# post processing
sys.path.append('../clip')
from clip import create_clip, recognize_from_image

logger = getLogger(__name__)

from bytetrack_utils import multiclass_nms
from tracker.byte_tracker import BYTETracker

# ======================
# Parameters
# ======================

WEIGHT_MOT17_X_PATH = 'bytetrack_x_mot17.onnx'
MODEL_MOT17_X_PATH = 'bytetrack_x_mot17.onnx.prototxt'
WEIGHT_MOT17_S_PATH = 'bytetrack_s_mot17.onnx'
MODEL_MOT17_S_PATH = 'bytetrack_s_mot17.onnx.prototxt'
WEIGHT_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx'
MODEL_MOT17_TINY_PATH = 'bytetrack_tiny_mot17.onnx.prototxt'
WEIGHT_MOT20_X_PATH = 'bytetrack_x_mot20.onnx'
MODEL_MOT20_X_PATH = 'bytetrack_x_mot20.onnx.prototxt'
REMOTE_PATH = \
    'https://storage.googleapis.com/ailia-models/bytetrack/'

WEIGHT_YOLOX_S_PATH = 'yolox_s.opt.onnx'
MODEL_YOLOX_S_PATH = 'yolox_s.opt.onnx.prototxt'
WEIGHT_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx'
MODEL_YOLOX_TINY_PATH = 'yolox_tiny.opt.onnx.prototxt'
REMOTE_YOLOX_PATH = \
    'https://storage.googleapis.com/ailia-models/yolox/'

VIDEO_PATH = 'demo.mp4'

IMAGE_MOT17_X_HEIGHT = 800
IMAGE_MOT17_X_WIDTH = 1440
IMAGE_MOT17_S_HEIGHT = 608
IMAGE_MOT17_S_WIDTH = 1088
IMAGE_MOT17_TINY_HEIGHT = 416
IMAGE_MOT17_TINY_WIDTH = 416
IMAGE_MOT20_X_HEIGHT = 896
IMAGE_MOT20_X_WIDTH = 1600
IMAGE_YOLOX_S_HEIGHT = 640
IMAGE_YOLOX_S_WIDTH = 640
IMAGE_YOLOX_TINY_HEIGHT = 416
IMAGE_YOLOX_TINY_WIDTH = 416

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ByteTrack', VIDEO_PATH, None
)
parser.add_argument(
    "--score_thre", type=float, default=0.1,
    help="Score threshould to filter the result.",
)
parser.add_argument(
    "--nms_thre", type=float, default=0.7,
    help="NMS threshould.",
)
parser.add_argument(
    '-m', '--model_type', default='mot17_s',
    choices=('mot17_x', 'mot20_x', 'mot17_s', 'mot17_tiny', 'yolox_s', 'yolox_tiny'),
    help='model type'
)
parser.add_argument(
    '--gui',
    action='store_true',
    help='Display preview in GUI.'
)
parser.add_argument(
    '--clip',
    action='store_true',
    help='Apply clip after detection.'
)
parser.add_argument(
    '-t', '--text', dest='text_inputs', type=str,
    action='append',
    help='Input text. (can be specified multiple times)'
)
parser.add_argument(
    '--crossing_line', type=str, default=None,
    help='Set crossing line x1 y1 x2 y2 x3 y3 x4 y4.'
)
parser.add_argument(
    '--csvpath', type=str, default=None,
    help='Set output csv.'
)
# tracking args
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
args = update_parser(parser)


# ======================
# Clip
# ======================

if args.text_inputs:
    clip_text = args.text_inputs
else:
    clip_text = ["man", "woman"]


# ======================
# Secondaty Functions
# ======================

def get_colors(n, colormap="gist_ncar"):
    # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
    # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
    # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
    # For more options see:
    # https://matplotlib.org/examples/color/colormaps_reference.html
    # and https://matplotlib.org/users/colormaps.html

    colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
    # Randomly shuffle the colors
    np.random.shuffle(colors)
    # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
    # Also multiply by 255 since cm returns values in the range [0, 1]
    colors = colors[:, (2, 1, 0)] * 255

    return colors


num_colors = 50
vis_colors = get_colors(num_colors)


def frame_vis_generator(frame, bboxes, ids):
    for i, entity_id in enumerate(ids):
        color = vis_colors[int(entity_id) % num_colors]

        x1, y1, w, h = np.round(bboxes[i]).astype(int)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)

    return frame

# ======================
# Line crossing
# ======================

target_lines = []
human_count_in = 0
human_count_out = 0

def intersect(p1, p2, p3, p4):
    tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    return tc1*tc2<0 and td1*td2<0

def display_line(frame):
    if len(target_lines) >= 4:
        cv2.line(frame, target_lines[2], target_lines[3], (255,0,0), thickness=5)
        cv2.putText(frame, "OUT", (target_lines[2][0] + 5,target_lines[2][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=3)
    if len(target_lines) >= 2:
        cv2.line(frame, target_lines[0], target_lines[1], (0,0,255), thickness=5)
        cv2.putText(frame, "IN", (target_lines[0][0] + 5,target_lines[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=3)
    for i in range(0, len(target_lines)):
        if i <= 1:
            color = (0,0,255)
        else:
            color = (255,0,0)
        cv2.circle(frame, center = target_lines[i], radius = 10, color=color, thickness=3)

TRACKING_STATE_NONE = 0
TRACKING_STATE_IN = 1
TRACKING_STATE_OUT = 2
TRACKING_STATE_DONE = 3

def line_crossing(frame, online_targets, tracking_position, tracking_state, tracking_guard, countup_state, frame_no, fps_time, total_time, net_clip, clip_id, clip_count):
    display_line(frame)

    global human_count_in, human_count_out
    for t in online_targets:
        # get one person
        tlwh = t.tlwh
        tid = t.track_id
        x = int(tlwh[0] + tlwh[2]/2)
        y = int(tlwh[1] + tlwh[3]/2)
        y_top = int(tlwh[1])
        if not (tid in tracking_position):
            tracking_position[tid] = []
            tracking_state[tid] = TRACKING_STATE_NONE
        tracking_position[tid].append({"x":x,"y":y,"frame_no":frame_no})

        # get history
        before = None
        line_before = None
        countup_in = False
        countup_out = False
        color = vis_colors[int(tid) % num_colors]
        original_color = color
        for data in tracking_position[tid]:
            if frame_no - data["frame_no"] >= 10:
                continue
            if before == None:
                before = data
                line_before = data
                continue
            if tracking_state[tid] == TRACKING_STATE_DONE:
                color = (0, 0, 0)
            cv2.line(frame, (before["x"],before["y"]), (data["x"], data["y"]), color, thickness=3)
            
            # detect line crossing
            if len(target_lines) >= 2:
                if intersect((line_before["x"],line_before["y"]), (data["x"], data["y"]), target_lines[0], target_lines[1]):
                    if tracking_state[tid] == TRACKING_STATE_OUT or tracking_state[tid] == TRACKING_STATE_DONE:
                        tracking_guard[tid] = frame_no
                        if tracking_state[tid] != TRACKING_STATE_DONE:
                            tracking_state[tid] = TRACKING_STATE_DONE
                            countup_in = True
                            human_count_in = human_count_in + 1
                    else:
                        tracking_state[tid] = TRACKING_STATE_IN
                if intersect((line_before["x"],line_before["y"]), (data["x"], data["y"]), target_lines[2], target_lines[3]):
                    if tracking_state[tid] == TRACKING_STATE_IN or tracking_state[tid] == TRACKING_STATE_DONE:
                        tracking_guard[tid] = frame_no
                        if tracking_state[tid] != TRACKING_STATE_DONE:
                            tracking_state[tid] = TRACKING_STATE_DONE
                            countup_out = True
                            human_count_out = human_count_out + 1
                    else:
                        tracking_state[tid] = TRACKING_STATE_OUT
            before = data

        # display id
        text = str(tid)
        cv2.putText(frame, text, (x, y_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=3)
        if tid in clip_id:
            cv2.putText(frame, clip_id[tid], (x, y_top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, original_color, thickness=3)

        # display detected person
        thickness = 0
        if tracking_state[tid] != TRACKING_STATE_NONE and tracking_state[tid] != TRACKING_STATE_DONE:
            thickness = 3
        if countup_in or countup_out:
            thickness = 10
            countup_state.append({"x":x,"y":y,"frame_no":frame_no})
        if thickness != 0:
            cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color=color, thickness=thickness)
            if args.clip and (countup_in or countup_out):
                img = frame[int(tlwh[1]):int(tlwh[1]+tlwh[3]), int(tlwh[0]):int(tlwh[0]+tlwh[2]),:]
                prob = recognize_from_image(net_clip, img)
                i = np.argmax(prob[0])
                clip_id[tid] = clip_text[i] + " " + str(int(prob[0][i]*100)/100)
                clip_count[i] = clip_count[i] + 1
        
        # recovery
        if tid in tracking_guard:
            if frame_no - tracking_guard[tid] >= 30:
                tracking_state[tid] = TRACKING_STATE_NONE
            
    for count in countup_state:
        t = frame_no - count["frame_no"]
        if t >= 10:
            continue
        t = (10 - t) * 4
        cv2.circle(frame, center = (count["x"],count["y"]), radius = t, color=(255,255,255), thickness=2)

    cv2.putText(frame, "Count(In) : " + str(human_count_in)+" Count(Out) : " + str(human_count_out)+" Time(sec) : "+str(fps_time)+ " / "+str(total_time), (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=3)


# ======================
# Csv output
# ======================

def open_csv():
    csv = open(args.csvpath, mode = 'w')
    csv.write("time(sec) , count(in) , count(out) , total_count(in) , total_count(out)")
    if args.clip:
        for i in range(0, len(clip_text)):
            csv.write(" , ")
            csv.write(clip_text[i])
    csv.write("\n")
    return csv


def write_csv(csv, fps_time, human_count_in, total_count_in, human_count_out, total_count_out, clip_count, total_clip_count):
    csv.write(str(fps_time) + " , " + str(human_count_in - total_count_in) + " , " +  str(human_count_out - total_count_out) + " , " + str(human_count_in) + " , " + str(human_count_out))
    if args.clip:
        for i in range(0, len(clip_text)):
            csv.write(" , ")
            csv.write(str(clip_count[i] - total_clip_count[i]))
    csv.write("\n")
    csv.flush()


# ======================
# Main functions
# ======================

def preprocess(img, img_size, normalize=True):
    h, w = img_size
    im_h, im_w, _ = img.shape

    r = min(h / im_h, w / im_w)
    oh, ow = int(im_h * r), int(im_w * r)

    resized_img = cv2.resize(
        img,
        (ow, oh),
        interpolation=cv2.INTER_LINEAR,
    )

    img = np.ones((h, w, 3)) * 114.0
    img[: oh, : ow] = resized_img

    if normalize:
        img = img[:, :, ::-1]  # BGR -> RGB
        img = normalize_image(img, 'ImageNet')

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, r


def postprocess(output, ratio, img_size, p6=False, nms_thre=0.7, score_thre=0.1):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    output[..., :2] = (output[..., :2] + grids) * expanded_strides
    output[..., 2:4] = np.exp(output[..., 2:4]) * expanded_strides

    predictions = output[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thre, score_thr=score_thre)

    return dets[:, :-1] if dets is not None else np.zeros((0, 5))


def predict(net, img):
    dic_model = {
        'mot17_x': (IMAGE_MOT17_X_HEIGHT, IMAGE_MOT17_X_WIDTH),
        'mot17_s': (IMAGE_MOT17_S_HEIGHT, IMAGE_MOT17_S_WIDTH),
        'mot17_tiny': (IMAGE_MOT17_TINY_HEIGHT, IMAGE_MOT17_TINY_WIDTH),
        'mot20_x': (IMAGE_MOT20_X_HEIGHT, IMAGE_MOT20_X_WIDTH),
        'yolox_s': (IMAGE_YOLOX_S_HEIGHT, IMAGE_YOLOX_S_WIDTH),
        'yolox_tiny': (IMAGE_YOLOX_TINY_HEIGHT, IMAGE_YOLOX_TINY_WIDTH),
    }
    model_type = args.model_type
    img_size = dic_model[model_type]

    img, ratio = preprocess(img, img_size, normalize=model_type.startswith('mot'))

    # feedforward
    output = net.predict([img])
    output = output[0]

    # For yolox, retrieve only the person class
    output = output[..., :6]

    score_thre = args.score_thre
    nms_thre = args.nms_thre
    dets = postprocess(output, ratio, img_size, nms_thre=nms_thre, score_thre=score_thre)

    return dets


def recognize_from_video(net, net_clip):
    min_box_area = args.min_box_area
    mot20 = args.model_type == 'mot20'

    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if args.savepath != None:
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    tracker = BYTETracker(
        track_thresh=args.track_thresh, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh, frame_rate=30,
        mot20=mot20)

    if args.csvpath != None:
        csv = open_csv()
    else:
        csv = None

    global target_lines
    if not args.crossing_line:
        m = f_w // 100
        target_lines.append( (f_w // 2 - m, 0) )
        target_lines.append( (f_w // 2 - m, f_h) )
        target_lines.append( (f_w // 2 + m, 0) )
        target_lines.append( (f_w // 2 + m, f_h) )
    else:
        texts= args.crossing_line.split(" ")
        target_lines = []
        target_lines.append( (int(texts[0]),int(texts[1])) )
        target_lines.append( (int(texts[2]),int(texts[3])) )
        target_lines.append( (int(texts[4]),int(texts[5])) )
        target_lines.append( (int(texts[6]),int(texts[7])) )

    frame_no = 0
    tracking_position = {}
    tracking_state = {}
    tracking_guard = {}
    clip_id = {}
    countup_state = []

    frame_shown = False
    before_fps_time = -1
    total_count_in = 0
    total_count_out = 0

    clip_count = []
    total_clip_count = []
    if args.clip:
        for i in range(0, len(clip_text)):
            clip_count.append(0)
            total_clip_count.append(0)

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        output = predict(net, frame)

        # run tracking
        online_targets = tracker.update(output)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

        # display
        fps_time = int(frame_no / fps)
        total_time = int(frames / fps)
        line_crossing(frame, online_targets, tracking_position, tracking_state, tracking_guard, countup_state, frame_no, fps_time, total_time, net_clip, clip_id, clip_count)
        res_img = frame

        #res_img = frame_vis_generator(frame, online_tlwhs, online_ids)

        # show
        if args.gui or args.video:
            cv2.imshow('frame', res_img)
            frame_shown = True
        else:
            print("Online ids",online_ids)

        # save results
        if writer is not None:
            writer.write(res_img.astype(np.uint8))
        if csv is not None:
            global human_count_in, human_count_out
            if before_fps_time != fps_time:
                write_csv(csv, fps_time, human_count_in, total_count_in, human_count_out, total_count_out, clip_count, total_clip_count)
                before_fps_time = fps_time
                total_count_in = human_count_in
                total_count_out = human_count_out
                if args.clip:
                    for i in range(0, len(clip_text)):
                        total_clip_count[i] = clip_count[i]

        frame_no = frame_no + 1

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    if csv is not None:
        csv.close()

    logger.info('Script finished successfully.')

# ======================
# MAIN functions
# ======================

def main():
    dic_model = {
        'mot17_x': (WEIGHT_MOT17_X_PATH, MODEL_MOT17_X_PATH),
        'mot17_s': (WEIGHT_MOT17_S_PATH, MODEL_MOT17_S_PATH),
        'mot17_tiny': (WEIGHT_MOT17_TINY_PATH, MODEL_MOT17_TINY_PATH),
        'mot20_x': (WEIGHT_MOT20_X_PATH, MODEL_MOT20_X_PATH),
        'yolox_s': (WEIGHT_YOLOX_S_PATH, MODEL_YOLOX_S_PATH),
        'yolox_tiny': (WEIGHT_YOLOX_TINY_PATH, MODEL_YOLOX_TINY_PATH),
    }
    model_type = args.model_type
    weight_path, model_path = dic_model[model_type]

    # model files check and download
    check_and_download_models(
        weight_path, model_path,
        REMOTE_PATH if model_type.startswith('mot') else REMOTE_YOLOX_PATH)

    env_id = args.env_id

    # initialize
    mem_mode = ailia.get_memory_mode(reduce_constant=True, reuse_interstage=True)
    net = ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=mem_mode)

    if args.clip:
        net_clip = create_clip(clip_text, args.env_id)
    else:
        net_clip = None

    recognize_from_video(net, net_clip)

if __name__ == '__main__':
    main()
