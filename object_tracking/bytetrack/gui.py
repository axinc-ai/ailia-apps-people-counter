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
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ByteTrack', None, None
)
parser.add_argument(
    '--csvpath', type=str, default=None,
    help='Set output csv.'
)
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
    global textInputVideoDetail
    fTyp = [("Image Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.video = file_name
        textInputVideoDetail.set(os.path.basename(args.video))

g_frame_shown = False

def close_crossing_line():
    textCrossingLine.set("Set crossing line")
    cv2.destroyAllWindows()
    g_frame_shown = False

def set_crossing_line():
    global g_frame, g_frame_shown
    global textCrossingLine

    if g_frame_shown:
        close_crossing_line()
        return

    capture = get_capture(args.video)
    assert capture.isOpened(), 'Cannot capture source'
    ret, frame = capture.read()
    g_frame = frame

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', set_line)

    frame = g_frame.copy()
    display_line(frame)

    cv2.imshow('frame', frame)
    
    textCrossingLine.set("Complete crossing line")
    g_frame_shown = True

def set_line(event,x,y,flags,param):
    global target_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(target_lines)>=2:
            target_lines = []
        target_lines.append((x,y))
        frame = g_frame.copy()
        display_line(frame)
        cv2.imshow('frame', frame)

def output_video_dialog():
    global textOutputVideoDetail
    fTyp = [("Output Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.savepath = file_name
        textOutputVideoDetail.set(os.path.basename(args.savepath))


def output_csv_dialog():
    global textOutputCsvDetail
    fTyp = [("Output Csv File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.csvpath = file_name
        textOutputCsvDetail.set(os.path.basename(args.csvpath))

root = None

def ui():
    # rootメインウィンドウの設定
    global root
    root = tk.Tk()
    root.title("ailia AI Analytics GUI")
    root.geometry("600x200")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=10,pady=10)

    textInputVideo = tk.StringVar(frame)
    textInputVideo.set("Input video")
    buttonInputVideo = tk.Button(frame, textvariable=textInputVideo, command=input_video_dialog, width=14)
    buttonInputVideo.grid(row=0, column=0, sticky=tk.NW)

    global textInputVideoDetail
    textInputVideoDetail = tk.StringVar(frame)
    textInputVideoDetail.set(args.video)
    labelInputVideoDetail = tk.Label(frame, textvariable=textInputVideoDetail)
    labelInputVideoDetail.grid(row=0, column=1, sticky=tk.NW)

    global textCrossingLine
    textCrossingLine = tk.StringVar(frame)
    textCrossingLine.set("Set crossing line")
    buttonCrossingLine = tk.Button(frame, textvariable=textCrossingLine, command=set_crossing_line, width=14)
    buttonCrossingLine.grid(row=1, column=0, sticky=tk.NW)

    textOutputVideo = tk.StringVar(frame)
    textOutputVideo.set("Output video")
    buttonOutputVideo = tk.Button(frame, textvariable=textOutputVideo, command=output_video_dialog, width=14)
    buttonOutputVideo.grid(row=2, column=0, sticky=tk.NW)

    global textOutputVideoDetail
    textOutputVideoDetail = tk.StringVar(frame)
    textOutputVideoDetail.set(args.savepath)
    labelOutputVideoDetail= tk.Label(frame, textvariable=textOutputVideoDetail)
    labelOutputVideoDetail.grid(row=2, column=1, sticky=tk.NW)

    textOutputCsv = tk.StringVar(frame)
    textOutputCsv.set("Output csv")
    buttonOutputCsv = tk.Button(frame, textvariable=textOutputCsv, command=output_csv_dialog, width=14)
    buttonOutputCsv.grid(row=3, column=0, sticky=tk.NW)

    global textOutputCsvDetail
    textOutputCsvDetail = tk.StringVar(frame)
    textOutputCsvDetail.set(args.csvpath)
    labelOutputCsvDetail= tk.Label(frame, textvariable=textOutputCsvDetail)
    labelOutputCsvDetail.grid(row=3, column=1, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Run")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=run, width=14)
    buttonTrainVideo.grid(row=4, column=0, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Stop")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=stop, width=14)
    buttonTrainVideo.grid(row=5, column=0, sticky=tk.NW)

    root.mainloop()

# ======================
# MAIN functions
# ======================

def main():
    args.video = "demo.mp4"
    args.savepath = "output.mp4"
    args.csvpath = "output.csv"
    ui()

import subprocess

proc = None

def run():
    close_crossing_line()

    global proc

    if not (proc==None):
        proc.kill()
        proc=None

    cmd = sys.executable

    args_dict = {}#vars(args)
    args_dict["video"] = args.video#"demo.mp4"
    if args.savepath:
        args_dict["savepath"] = args.savepath
    if args.csvpath:
        args_dict["csvpath"] = args.csvpath
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

def stop():
    global proc

    if not (proc==None):
        proc.kill()
        proc=None

if __name__ == '__main__':
    main()
