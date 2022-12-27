import sys
import time

import numpy as np
import cv2
from matplotlib import cm
from PIL import Image, ImageTk

import ailia

# import original modules
sys.path.append('./util')
from utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import os

logger = getLogger(__name__)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ailia APPS people counter', None, None
)
parser.add_argument(
    '--camera', type=str, default=None,
    help='Set web cam mode.'
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
# Video
# ======================

def input_video_dialog():
    global textInputVideoDetail
    fTyp = [("All Files", "*.*"), ("Video files","*.mp4")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.video = file_name
        textInputVideoDetail.set(os.path.basename(args.video))

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

# ======================
# Environment
# ======================

env_index = args.env_id

def get_env_list():
    env_list = []
    for env in ailia.get_environment_list():
        env_list.append(env.name)
    return env_list  

def environment_changed(event):
    global env_index
    selection = event.widget.curselection()
    if selection:
        env_index = selection[0]
    else:
        env_index = 0
    #print("env",env_index)

# ======================
# Model
# ======================

model_index = 0

def get_model_list():
    model_list = ["mot17_s", "mot17_tiny"]
    return model_list  

def model_changed(event):
    global model_index
    selection = event.widget.curselection()
    if selection:
        model_index = selection[0]
    else:
        model_index = 0
    #print("model",model_index)

# ======================
# Line crossing
# ======================

target_lines = []
human_count = 0

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

g_frame = None
crossingLineWindow = None

def close_crossing_line():
    global crossingLineWindow
    if crossingLineWindow != None and crossingLineWindow.winfo_exists():
        crossingLineWindow.destroy()
        crossingLineWindow = None

def set_crossing_line():
    global g_frame, g_frame_shown
    global textCrossingLine
    global crossingLineWindow

    if crossingLineWindow != None and crossingLineWindow.winfo_exists():
        return

    capture = get_capture(args.video)
    assert capture.isOpened(), 'Cannot capture source'
    ret, frame = capture.read()
    g_frame = frame

    crossingLineWindow = tk.Toplevel()
    crossingLineWindow.title("Set crossing line")
    crossingLineWindow.geometry(str(g_frame.shape[1])+"x"+str(g_frame.shape[0]))
    tk.Label(crossingLineWindow, text ="Please set crossing line by click").pack()
    crossingLineWindow.canvas = tk.Canvas(crossingLineWindow)
    crossingLineWindow.canvas.bind('<Button-1>', set_line)
    crossingLineWindow.canvas.pack(expand = True, fill = tk.BOTH)

    frame = g_frame.copy()
    display_line(frame)
    update_frame_image(frame)

def update_frame_image(frame):
    global crossingLineWindow
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    crossingLineWindow.photo_image = ImageTk.PhotoImage(image=pil_image)
    crossingLineWindow.canvas.create_image(
            frame.shape[1] / 2,
            frame.shape[0] / 2,                   
            image=crossingLineWindow.photo_image
            )

def set_line(event):
    global target_lines
    x = event.x
    y = event.y
    if len(target_lines)>=4:
        target_lines = []
    target_lines.append((x,y))
    frame = g_frame.copy()
    display_line(frame)
    update_frame_image(frame)

# ======================
# GUI functions
# ======================

root = None
checkBoxClipBln = None
checkBoxAgeGenderBln = None
clipTextEntery = None

def ui():
    # rootメインウィンドウの設定
    global root
    root = tk.Tk()
    root.title("ailia APPS People Counter")
    root.geometry("600x300")

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=10,pady=10)

    textInputVideo = tk.StringVar(frame)
    textInputVideo.set("Input video")
    buttonInputVideo = tk.Button(frame, textvariable=textInputVideo, command=input_video_dialog, width=14)
    buttonInputVideo.grid(row=0, column=0, sticky=tk.NW)

    global textInputVideoDetail
    textInputVideoDetail = tk.StringVar(frame)
    if args.camera:
        textInputVideoDetail.set("camera "+args.camera)
    else:
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

    model_list = get_model_list()
    env_list = get_env_list()

    lists = tk.StringVar(value=model_list)
    listEnvironment =tk.StringVar(value=env_list)

    ListboxModel = tk.Listbox(frame, listvariable=lists, width=40, height=3, selectmode="single", exportselection=False)
    ListboxEnvironment = tk.Listbox(frame, listvariable=listEnvironment, width=40, height=4, selectmode="single", exportselection=False)

    ListboxModel.bind("<<ListboxSelect>>", model_changed)
    ListboxEnvironment.bind("<<ListboxSelect>>", environment_changed)

    ListboxModel.select_set(model_index)
    ListboxEnvironment.select_set(env_index)

    textModel = tk.StringVar(frame)
    textModel.set("Models")
    labelModel = tk.Label(frame, textvariable=textModel)
    labelModel.grid(row=0, column=2, sticky=tk.NW, rowspan=1)
    ListboxModel.grid(row=1, column=2, sticky=tk.NW, rowspan=2)

    textEnvironment = tk.StringVar(frame)
    textEnvironment.set("Environment")
    labelEnvironment = tk.Label(frame, textvariable=textEnvironment)
    labelEnvironment.grid(row=3, column=2, sticky=tk.NW, rowspan=1)
    ListboxEnvironment.grid(row=4, column=2, sticky=tk.NW, rowspan=4)

    global checkBoxClipBln
    checkBoxClipBln = tkinter.BooleanVar()
    checkBoxClipBln.set(False)
    checkBoxClip = tkinter.Checkbutton(frame, variable=checkBoxClipBln, text='Clip classification')
    checkBoxClip.grid(row=8, column=2, sticky=tk.NW, rowspan=1)

    global clipTextEntery
    clipTextEntery = tkinter.Entry(frame, width=20)
    clipTextEntery.insert(tkinter.END,"man,woman")
    clipTextEntery.grid(row=9, column=2, sticky=tk.NW, rowspan=1)

    global checkBoxAgeGenderBln
    checkBoxAgeGenderBln = tkinter.BooleanVar()
    checkBoxAgeGenderBln.set(False)
    checkBoxAgeGender = tkinter.Checkbutton(frame, variable=checkBoxAgeGenderBln, text='Age gender classification')
    checkBoxAgeGender.grid(row=10, column=2, sticky=tk.NW, rowspan=1)

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
    if args.camera:
        args_dict["video"] = args.camera
    else:
        args_dict["video"] = args.video
    if args.savepath:
        args_dict["savepath"] = args.savepath
    if args.csvpath:
        args_dict["csvpath"] = args.csvpath
    
    global model_index
    args_dict["model_type"] = get_model_list()[model_index]

    global env_index
    args_dict["env_id"] = env_index

    global checkBoxClipBln
    if checkBoxClipBln.get():
        args_dict["clip"] = True

    global checkBoxAgeGenderBln
    if checkBoxAgeGenderBln.get():
        args_dict["age_gender"] = True

    if len(target_lines) >= 4:
        line1 = str(target_lines[0][0]) + " " + str(target_lines[0][1]) + " " + str(target_lines[1][0]) + " " + str(target_lines[1][1])
        line2 = str(target_lines[2][0]) + " " + str(target_lines[2][1]) + " " + str(target_lines[3][0]) + " " + str(target_lines[3][1])
        args_dict["crossing_line"] = line1 + " " + line2

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

    global clipTextEntery
    if clipTextEntery:
        clip_text = clipTextEntery.get().split(",")
        for text in clip_text:
                options.append("--text")
                options.append(text)#"\""+text+"\"")

    cmd = [cmd, "bytetrack.py"] + options
    print(" ".join(cmd))

    dir = "./object_tracking/bytetrack/"

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
