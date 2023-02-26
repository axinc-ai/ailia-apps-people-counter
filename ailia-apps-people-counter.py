# ailia APPS People Counter
# (C) 2022-2023 AXELL CORPORATION

import sys
import time

import numpy as np
import cv2
from matplotlib import cm
from PIL import Image, ImageTk

import ailia
import json

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
import tkinter.messagebox
import os

logger = getLogger(__name__)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ailia APPS people counter', None, None
)
args = update_parser(parser)


# ======================
# Input Video
# ======================

input_index = 0
listsInput = None
ListboxInput = None
input_list = []

def get_input_list():
    if args.debug:
        return ["Camera:0"]

    index = 0
    inputs = []
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            inputs.append("Camera:"+str(index))
        else:
            break
        index=index+1
        cap.release()

    if len(inputs) == 0:
        inputs.append("demo.mp4")

    return inputs

def input_changed(event):
    global input_index, input_list, textInputVideoDetail
    selection = event.widget.curselection()
    if selection:
        input_index = selection[0]
    else:
        input_index = 0   
    if "Camera:" in input_list[input_index]:
        textInputVideoDetail.set(input_list[input_index])
    else:
        textInputVideoDetail.set(os.path.basename(input_list[input_index]))
        
    #print("input",input_index)

def input_video_dialog():
    global textInputVideoDetail, listsInput, ListboxInput, input_index, input_list
    fTyp = [("All Files", "*.*"), ("Video files","*.mp4")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        textInputVideoDetail.set(os.path.basename(file_name))
        input_list.append(file_name)
        listsInput.set(input_list)
        ListboxInput.select_clear(input_index)
        input_index = len(input_list)-1
        ListboxInput.select_set(input_index)

# ======================
# Output Video, Csv, Image
# ======================

def apply_path_to_ui():
    global textOutputVideoDetail
    textOutputVideoDetail.set(os.path.basename(args.savepath))
    global textOutputCsvDetail
    textOutputCsvDetail.set(os.path.basename(args.csvpath))
    global textOutputImageDetail
    textOutputImageDetail.set(os.path.basename(args.imgpath))


def output_video_dialog():
    fTyp = [("Output Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.savepath = file_name
        apply_path_to_ui()


def output_csv_dialog():
    fTyp = [("Output Csv File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.csvpath = file_name
        apply_path_to_ui()


def output_img_dialog():
    fTyp = [("Output Image Folder", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askdirectory(initialdir=iDir)
    if len(file_name) != 0:
        args.imgpath = file_name
        apply_path_to_ui()

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
    model_list = ["mot17_s", "mot17_tiny", "yolox_s"]
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
# Category
# ======================

category_index = 0

def get_category_list():
    category_list = ["person", "vehicle"]
    return category_list  

def category_changed(event):
    global category_index
    selection = event.widget.curselection()
    if selection:
        category_index = selection[0]
    else:
        category_index = 0

# ======================
# Line crossing
# ======================

target_lines = [
    {"id":"line0","lines":[(0,0),(100,0),(100,100),(0,100)]}
]
human_count = 0

def display_line(frame):
    for id in range(len(target_lines)):
        lines = target_lines[id]["lines"]
        if len(lines) >= 4:
            cv2.line(frame, (lines[2][0], lines[2][1]), (lines[3][0], lines[3][1]), (255,0,0), thickness=5)
            cv2.putText(frame, "OUT", (lines[2][0] + 5,lines[2][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=3)
        if len(lines) >= 2:
            cv2.line(frame, (lines[0][0], lines[0][1]) , (lines[1][0], lines[1][1]), (0,0,255), thickness=5)
            cv2.putText(frame, "IN", (lines[0][0] + 5,lines[0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=3)
        for i in range(0, len(lines)):
            if i <= 1:
                color = (0,0,255)
            else:
                color = (255,0,0)
            cv2.circle(frame, center = (lines[i][0], lines[i][1]), radius = 10, color=color, thickness=3)

g_frame = None
crossingLineWindow = None

def close_crossing_line():
    global crossingLineWindow
    if crossingLineWindow != None and crossingLineWindow.winfo_exists():
        crossingLineWindow.destroy()
        crossingLineWindow = None

def get_video_path():
    global input_list, input_index
    if "Camera:" in input_list[input_index]:
        return input_index
    else:
        return input_list[input_index]

def set_crossing_line():
    global g_frame, g_frame_shown
    global textCrossingLine
    global crossingLineWindow

    if crossingLineWindow != None and crossingLineWindow.winfo_exists():
        return

    capture = get_capture(get_video_path())
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
    lines = target_lines[0]["lines"]
    if len(lines)>=4:
        lines.clear()
    else:
        lines.append((x,y))
    frame = g_frame.copy()
    display_line(frame)
    update_frame_image(frame)

# ======================
# Menu functions
# ======================

api_secret = ""
measurement_id = ""

def get_settings():
    global target_lines
    settings = {}
    settings["target_lines"] = target_lines

    global model_index
    settings["model_type"] = get_model_list()[model_index]

    global category_index
    settings["category"] = get_category_list()[category_index]

    global clipTextEntry
    settings["clip_text"] = clipTextEntry.get()

    global checkBoxAgeGenderBln
    if checkBoxAgeGenderBln.get():
        settings["age_gender"] = True
    else:
        settings["age_gender"] = False

    global checkBoxClipBln
    if checkBoxClipBln.get():
        settings["clip"] = True
    else:
        settings["clip"] = False

    global checkBoxAlwaysBln
    if checkBoxAlwaysBln.get():
        settings["always_classify_for_debug"] = True
    else:
        settings["always_classify_for_debug"] = False

    global api_secret, measurement_id
    settings["api_secret"] = api_secret
    settings["measurement_id"] = measurement_id

    settings["savepath"] = args.savepath
    settings["csvpath"] = args.csvpath
    settings["imgpath"] = args.imgpath
    
    return settings

def search_list_idx(model_list, model_id):
    for i in range(len(model_list)):
        if model_id == model_list[i]:
            return i
    return 0

def set_settings(settings):
    global target_lines
    target_lines = settings["target_lines"]

    global model_index, ListboxModel
    ListboxModel.select_clear(model_index)
    model_index = search_list_idx(get_model_list(), settings["model_type"])
    ListboxModel.select_set(model_index)

    if "category" in settings:
        global category_index, ListboxCategory
        ListboxCategory.select_clear(category_index)
        category_index = search_list_idx(get_category_list(), settings["category"])
        ListboxCategory.select_set(category_index)

    global clipTextEntry
    clipTextEntry.delete(0, tk.END)
    clipTextEntry.insert(0, str(settings["clip_text"]))

    global checkBoxAgeGenderBln
    checkBoxAgeGenderBln.set(settings["age_gender"])

    global checkBoxClipBln
    checkBoxClipBln.set(settings["clip"])

    global checkBoxAlwaysBln
    checkBoxAlwaysBln.set(settings["always_classify_for_debug"])

    global api_secret
    if "api_secret" in settings:
        api_secret = settings["api_secret"]
        global apiSecretEntry
        if apiSecretEntry != None:
            apiSecretEntry.set(api_secret)

    global measurement_id
    if "measurement_id" in settings:
        measurement_id = settings["measurement_id"]
        global measurementIdEntry
        if measurementIdEntry != None:
            measurementIdEntry.set(measurement_id)
    
    if "savepath" in settings:
        args.savepath = settings["savepath"]
    if "csvpath" in settings:
        args.csvpath = settings["csvpath"]
    if "imgpath" in settings:
        args.imgpath = settings["imgpath"]
    
    apply_path_to_ui()

def menu_file_open_click():
    fTyp = [("Config files","*.json")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        with open(file_name, 'r') as json_file:
            settings = json.load(json_file)
            set_settings(settings)

def menu_file_saveas_click():
    fTyp = [("Config files", "*.json")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        with open(file_name, 'w') as json_file:
            settings = get_settings()
            json.dump(settings, json_file)

analyticsWindow = None
apiSecretEntry = None
measurementIdEntry = None

def menu_analytics_open():
    global analyticsWindow

    if analyticsWindow != None and analyticsWindow.winfo_exists():
        return

    analyticsWindow = tk.Toplevel()
    analyticsWindow.title("Google Analytics Settings")
    analyticsWindow.geometry("300x300")

    frame = ttk.Frame(analyticsWindow)
    frame.pack(padx=10,pady=10)

    textOptions = tk.StringVar(frame)
    textOptions.set("API_SECRET")
    labelOptions = tk.Label(frame, textvariable=textOptions)
    labelOptions.grid(row=0, column=0, sticky=tk.NW)

    global apiSecretEntry
    apiSecretEntry = tkinter.Entry(frame, width=20)
    apiSecretEntry.insert(tkinter.END,api_secret)
    apiSecretEntry.grid(row=1, column=0, sticky=tk.NW, rowspan=1)

    textOptions = tk.StringVar(frame)
    textOptions.set("MEASUREMENT_ID")
    labelOptions = tk.Label(frame, textvariable=textOptions)
    labelOptions.grid(row=2, column=0, sticky=tk.NW)

    global measurementIdEntry
    measurementIdEntry = tkinter.Entry(frame, width=20)
    measurementIdEntry.insert(tkinter.END,measurement_id)
    measurementIdEntry.grid(row=3, column=0, sticky=tk.NW, rowspan=1)

    setAnalyticsSettingsText = tk.StringVar(frame)
    setAnalyticsSettingsText.set("OK")
    buttonSetAnalyticsSettings = tk.Button(frame, textvariable=setAnalyticsSettingsText, command=menu_analytics_set, width=14)
    buttonSetAnalyticsSettings.grid(row=4, column=0, sticky=tk.NW)

def menu_analytics_set():
    global apiSecretEntry, measurementIdEntry
    global api_secret, measurement_id
    api_secret = apiSecretEntry.get()
    measurement_id = measurementIdEntry.get()
    global analyticsWindow
    analyticsWindow.destroy()
    analyticsWindow = None

def menu(root):
    menubar = tk.Menu(root)

    menu_file = tk.Menu(menubar, tearoff = False)
    menu_file.add_command(label = "Load settings",  command = menu_file_open_click,  accelerator="Ctrl+O")
    menu_file.add_command(label = "Save settings", command = menu_file_saveas_click, accelerator="Ctrl+S")
    menu_file.add_separator() # 仕切り線
    menu_file.add_command(label = "Quit",            command = root.destroy)

    menu_analytics = tk.Menu(menubar, tearoff = False)
    menu_analytics.add_command(label = "Analytics settings",  command = menu_analytics_open,  accelerator="Ctrl+A")

    menubar.add_cascade(label="File", menu=menu_file)
    menubar.add_cascade(label="Analytics", menu=menu_analytics)

    root.config(menu=menubar)

# ======================
# GUI functions
# ======================

root = None
checkBoxClipBln = None
checkBoxAgeGenderBln = None
clipTextEntry = None
checkBoxAlwaysBln = None
ListboxModel = None

def ui():
    # rootメインウィンドウの設定
    global root
    root = tk.Tk()
    root.title("ailia APPS People Counter")
    root.geometry("720x360")

    # メニュー作成
    menu(root)

    # 環境情報取得
    global input_list
    input_list = get_input_list()
    model_list = get_model_list()
    env_list = get_env_list()

    # メインフレームの作成と設置
    frame = ttk.Frame(root)
    frame.pack(padx=10,pady=10)

    textInputVideo = tk.StringVar(frame)
    textInputVideo.set("Input video")
    buttonInputVideo = tk.Button(frame, textvariable=textInputVideo, command=input_video_dialog, width=14)
    buttonInputVideo.grid(row=0, column=0, sticky=tk.NW)

    global textInputVideoDetail
    textInputVideoDetail = tk.StringVar(frame)
    textInputVideoDetail.set(input_list[input_index])
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

    textOutputImage = tk.StringVar(frame)
    textOutputImage.set("Output image")
    buttonOutputImage = tk.Button(frame, textvariable=textOutputImage, command=output_img_dialog, width=14)
    buttonOutputImage.grid(row=4, column=0, sticky=tk.NW)

    global textOutputImageDetail
    textOutputImageDetail = tk.StringVar(frame)
    textOutputImageDetail.set(args.imgpath)
    labelOutputImageDetail= tk.Label(frame, textvariable=textOutputImageDetail)
    labelOutputImageDetail.grid(row=4, column=1, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Run")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=run, width=14)
    buttonTrainVideo.grid(row=5, column=0, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Stop")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=stop, width=14)
    buttonTrainVideo.grid(row=6, column=0, sticky=tk.NW)

    global listsInput, ListboxInput, ListboxModel

    textInputVideoHeader = tk.StringVar(frame)
    textInputVideoHeader.set("Inputs")
    labelInputVideoHeader = tk.Label(frame, textvariable=textInputVideoHeader)
    labelInputVideoHeader.grid(row=0, column=2, sticky=tk.NW)

    listsInput = tk.StringVar(value=input_list)
    ListboxInput = tk.Listbox(frame, listvariable=listsInput, width=26, height=4, selectmode="single", exportselection=False)
    ListboxInput.bind("<<ListboxSelect>>", input_changed)
    ListboxInput.select_set(input_index)
    ListboxInput.grid(row=1, column=2, sticky=tk.NW, rowspan=3, columnspan=2)

    lists = tk.StringVar(value=model_list)
    listEnvironment =tk.StringVar(value=env_list)

    ListboxModel = tk.Listbox(frame, listvariable=lists, width=26, height=3, selectmode="single", exportselection=False)
    ListboxEnvironment = tk.Listbox(frame, listvariable=listEnvironment, width=26, height=4, selectmode="single", exportselection=False)

    ListboxModel.bind("<<ListboxSelect>>", model_changed)
    ListboxEnvironment.bind("<<ListboxSelect>>", environment_changed)

    ListboxModel.select_set(model_index)
    ListboxEnvironment.select_set(env_index)

    textModel = tk.StringVar(frame)
    textModel.set("Models")
    labelModel = tk.Label(frame, textvariable=textModel)
    labelModel.grid(row=4, column=2, sticky=tk.NW, rowspan=1)
    ListboxModel.grid(row=5, column=2, sticky=tk.NW, rowspan=2)

    textEnvironment = tk.StringVar(frame)
    textEnvironment.set("Environment")
    labelEnvironment = tk.Label(frame, textvariable=textEnvironment)
    labelEnvironment.grid(row=8, column=2, sticky=tk.NW, rowspan=1)
    ListboxEnvironment.grid(row=9, column=2, sticky=tk.NW, rowspan=4)

    textOptions = tk.StringVar(frame)
    textOptions.set("Options")
    labelOptions = tk.Label(frame, textvariable=textOptions)
    labelOptions.grid(row=0, column=3, sticky=tk.NW)

    global checkBoxClipBln
    checkBoxClipBln = tkinter.BooleanVar()
    checkBoxClipBln.set(False)
    checkBoxClip = tkinter.Checkbutton(frame, variable=checkBoxClipBln, text='Clip classification')
    checkBoxClip.grid(row=1, column=3, sticky=tk.NW, rowspan=1)

    global clipTextEntry
    clipTextEntry = tkinter.Entry(frame, width=20)
    clipTextEntry.insert(tkinter.END,"man,woman")
    clipTextEntry.grid(row=2, column=3, sticky=tk.NW, rowspan=1)

    global checkBoxAgeGenderBln
    checkBoxAgeGenderBln = tkinter.BooleanVar()
    checkBoxAgeGenderBln.set(False)
    checkBoxAgeGender = tkinter.Checkbutton(frame, variable=checkBoxAgeGenderBln, text='Age gender classification')
    checkBoxAgeGender.grid(row=3, column=3, sticky=tk.NW, rowspan=1)

    global checkBoxAlwaysBln
    checkBoxAlwaysBln = tkinter.BooleanVar()
    checkBoxAlwaysBln.set(False)
    checkBoxAlways = tkinter.Checkbutton(frame, variable=checkBoxAlwaysBln, text='Always classify for debug')
    checkBoxAlways.grid(row=4, column=3, sticky=tk.NW, rowspan=1)

    textCategory = tk.StringVar(frame)
    textCategory.set("Category")
    labelCategory = tk.Label(frame, textvariable=textCategory)
    labelCategory.grid(row=5, column=3, sticky=tk.NW)

    global ListboxCategory
    lists = tk.StringVar(value=get_category_list())
    ListboxCategory = tk.Listbox(frame, listvariable=lists, width=26, height=3, selectmode="single", exportselection=False)
    ListboxCategory.grid(row=6, column=3, sticky=tk.NW, rowspan=2)
    ListboxCategory.bind("<<ListboxSelect>>", category_changed)

    ListboxCategory.select_set(category_index)

    root.mainloop()

# ======================
# MAIN functions
# ======================

def main():
    args.savepath = ""
    args.csvpath = ""
    args.imgpath = ""
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
    args_dict["video"] = get_video_path()
          
    global env_index
    args_dict["env_id"] = env_index

    settings = get_settings()
    if settings["savepath"]:
        args_dict["savepath"] = settings["savepath"]
    if settings["csvpath"]:
        args_dict["csvpath"] = settings["csvpath"]
    if settings["imgpath"]:
        args_dict["imgpath"] = settings["imgpath"]
    args_dict["clip"] = settings["clip"]
    args_dict["age_gender"] = settings["age_gender"]
    args_dict["always_classification"] = settings["always_classify_for_debug"]
    args_dict["model_type"] = settings["model_type"]
    args_dict["category"] = settings["category"]
    if settings["api_secret"] != "" and settings["measurement_id"] != "":
        args_dict["analytics_api_secret"] = settings["api_secret"]
        args_dict["analytics_measurement_id"] = settings["measurement_id"]

    if settings["category"] != "person" and (not "yolo" in settings["model_type"]):
        tk.messagebox.showerror("Model type error", "Please select yolo model for vehicle detection.")
        return

    crossing_line = ""
    for i in range(len(target_lines)):
        if (len(target_lines[i]["lines"]) >= 4):
            lines = target_lines[i]["lines"]
            line_id = target_lines[i]["id"]
            line1 = str(lines[0][0]) + " " + str(lines[0][1]) + " " + str(lines[1][0]) + " " + str(lines[1][1])
            line2 = str(lines[2][0]) + " " + str(lines[2][1]) + " " + str(lines[3][0]) + " " + str(lines[3][1])
            if crossing_line != "":
                crossing_line = crossing_line + " "
            crossing_line = crossing_line + line_id + " " + line1 + " " + line2
    if crossing_line != "":
        args_dict["crossing_line"] = crossing_line

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

    global clipTextEntry
    if clipTextEntry:
        clip_text = clipTextEntry.get().split(",")
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
