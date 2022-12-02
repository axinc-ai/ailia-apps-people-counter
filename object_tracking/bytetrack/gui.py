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
# tracking args
parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
args = update_parser(parser)


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
human_count = 0

def display_line(frame):
    if len(target_lines) >= 2:
        cv2.line(frame, target_lines[0], target_lines[1], (0,0,255), thickness=5)
    for i in range(0, len(target_lines)):
        cv2.circle(frame, center = target_lines[i], radius = 10, color=(0,0,255), thickness=3)

# ======================
# GUI functions
# ======================

import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import os

g_frame = None

def input_video_dialog():
    fTyp = [("Image Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.video = file_name

def set_crossing_line():
    global g_frame

    capture = get_capture(args.video)
    assert capture.isOpened(), 'Cannot capture source'
    ret, frame = capture.read()
    g_frame = frame

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', set_line)

    cv2.imshow('frame', g_frame)

def output_video_dialog():
    fTyp = [("Output Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        g_file_path = file_name

def output_csv_dialog():
    fTyp = [("Output Csv File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        g_file_path = file_name

def set_line(event,x,y,flags,param):
    global target_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(target_lines)>=2:
            target_lines = []
        target_lines.append((x,y))
        frame = g_frame.copy()
        display_line(frame)
        cv2.imshow('frame', frame)

def ui():
    # rootメインウィンドウの設定
    root = tk.Tk()
    root.title("ailia AI Analytics GUI")
    root.geometry("600x200")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=10,pady=10)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Input video")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=input_video_dialog, width=14)
    buttonTrainVideo.grid(row=0, column=0, sticky=tk.NW)

    textTrainVideo2 = tk.StringVar(frame)
    textTrainVideo2.set(args.video)
    labelInput = tk.Label(frame, textvariable=textTrainVideo2)
    labelInput.grid(row=0, column=1, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Set crossing line")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=set_crossing_line, width=14)
    buttonTrainVideo.grid(row=1, column=0, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Output video")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=output_video_dialog, width=14)
    buttonTrainVideo.grid(row=2, column=0, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Output csv")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=output_csv_dialog, width=14)
    buttonTrainVideo.grid(row=3, column=0, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Run")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=run, width=14)
    buttonTrainVideo.grid(row=4, column=0, sticky=tk.NW)

    root.mainloop()

# ======================
# MAIN functions
# ======================

def main():
    ui()

import subprocess

proc = None

def run():
    global proc

    if not (proc==None):
        proc.kill()
        proc=None

    cmd = sys.executable

    args_dict = {}#vars(args)
    args_dict["video"] = args.video#"demo.mp4"
    if len(target_lines) >= 2:
        args_dict["crossing_line"] = str(target_lines[0][0]) + " " + str(target_lines[0][1]) + " " + str(target_lines[1][0]) + " " + str(target_lines[1][1])

    options = []
    for key in args_dict:
        if key=="ftype":
            continue
        if args_dict[key] is not None:
            if args_dict[key] is True:
                options.append("--"+key)
            elif args_dict[key] is False:
                continue
            else:
                options.append("--"+key)
                options.append(str(args_dict[key]))

    cmd = [cmd, "bytetrack.py"] + options
    print(" ".join(cmd))

    dir = "./"

    proc = subprocess.Popen(cmd, cwd=dir)
    try:
        outs, errs = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        pass


if __name__ == '__main__':
    main()
