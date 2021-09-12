from os import path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pygame
import pyaudio
import threading
import torch
import collections
import struct
import time
import re
import wandb
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# from PIL import Image, ImageTk

import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

WORKING_DIR = os.path.join('.')
ENTITY = "thasthika"
PROJECT = "mer"

api = wandb.Api()

def __get_model_info(fp):
    _cn = list(filter(lambda x: x.startswith("class"),
               open(fp, mode="r").readlines()))[0]
    x = re.search("class ([^(_V)]+)_V([\d+])", _cn)
    model_version = int(x.group(2))
    model_name = x.group(1)
    return {'name': model_name, 'version': model_version}

def __load_model_class(run, model_version):

    runp = run.split(".")
    if len(runp) > 3:
        runp = runp[:-1]

    model_file = "model_v{}.py".format(model_version)
    run_path = path.join(WORKING_DIR, "models", path.join(*runp), model_file)
    model_info = __get_model_info(run_path)
    print(model_info)
    ModelClsName = "{}_V{}".format(model_info['name'], model_info['version'])

    runp = ".".join(runp)
    pkg_path = "models.{}.model_v{}".format(runp, model_version)

    print("Loading Model Class {} from {}".format(ModelClsName, pkg_path))

    modelMod = __import__(pkg_path, fromlist=[ModelClsName])
    ModelClass = getattr(modelMod, ModelClsName)

    return (ModelClass, model_info)

def __get_checkpoint_from_file(run):
    ckpt_file = None
    for f in run.files():
        if f.name.endswith(".ckpt"):
            try:
                ckpt_file = f.download()
                ckpt_file = ckpt_file.name
            except Exception as e:
                print(e)
                ckpt_file = "./{}".format(f.name)
            break
    return ckpt_file

def __get_checkpoint_from_artifact(run):
    ckpt_file = None
    for x in run.logged_artifacts():
        artifact_dir = x.download()
        files = os.listdir(artifact_dir)
        for f in files:
            if f.endswith(".ckpt"):
                ckpt_file = path.join(artifact_dir, f)
                return ckpt_file
    return ckpt_file

def get_model(run_name, run_id, version):
    run = api.run("{}/{}/{}".format(ENTITY, PROJECT, run_id))

    ckpt_file = __get_checkpoint_from_file(run)
    if ckpt_file is None:
        ckpt_file = __get_checkpoint_from_artifact(run)
    if ckpt_file is None:
        print("Warning!! Could not get the checkpoint file")

    config = run.config
    
    ModelClass, _ = __load_model_class(run_name, version)
    model = ModelClass(**config)
    
    if not ckpt_file is None:
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    return model

CHUNK = 1024 * 2
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
BUFFER = RATE * 5

WORK_DIR = path.dirname(__file__)
RES_DIR = path.join(WORK_DIR, "resources")

def get_resource_path(resource_file):
    return path.join(RES_DIR, resource_file)

device = "cuda" if torch.cuda.is_available() else "cpu"

inference_lock = threading.Lock()
inference_should_run = False

microphone_stop_lock = threading.Lock()
microphone_running = False
microphone_buffer = collections.deque(maxlen=BUFFER)
result_buffer = []
model = None
model_type = "categorical"
result_holder = None


def handle_audio_chunk(in_data, frame_count, time_info, status):
    global microphone_running
    global microphone_buffer

    global inference_lock
    global inference_should_run

    if microphone_running == False:
        inference_lock.acquire()
        inference_should_run = False
        inference_lock.release()
        return (None, pyaudio.paAbort)

    data = struct.unpack(str(CHUNK)+'f', in_data)
    microphone_buffer.extend(data)
    if len(microphone_buffer) < BUFFER:
        print("Gathering Data...")
        return (None, pyaudio.paContinue)


    return (None, pyaudio.paContinue)

def inference_run():
    global model
    global model_type
    global microphone_buffer
    global inference_should_run
    global inference_lock

    inference_lock.acquire()
    inference_should_run = True
    inference_lock.release()

    while True:
        time.sleep(.2)
        if not inference_should_run:
            break

        if model is None:
            continue

        if len(microphone_buffer) < BUFFER:
            continue
        
        # print("Inference Run...")

        buff = list(microphone_buffer)
        inp = torch.tensor(buff, dtype=torch.float32)

        (old_max_val, old_min_val) = torch.max(inp).item(), torch.min(inp).item()
        (new_max_val, new_min_val) = +1.0, -1.0

        old_range = (old_max_val - old_min_val)
        new_range = (new_max_val - new_min_val)
        inp = (((inp - old_min_val) * new_range) / old_range) + new_min_val

        inp = torch.reshape(inp, (1, 1, BUFFER))
        ret = model(inp)
        if model_type == "categorical":
            ret = torch.softmax(model(inp), dim=1)

        if result_holder is None:
            continue

        result_holder.set_value(ret)
        # print("Power: {}".format(torch.sum(torch.sqrt(torch.pow(torch.abs(inp), 2))) / BUFFER))

        # if not model:
        #     continue
        # inp = torch.tensor(list(microphone_buffer), dtype=torch.long).reshape((1, 1, int(BUFFER)))
        # print(inp)

def listen_microphone():
    global microphone_running
    global microphone_stop_lock

    global inference_lock
    global inference_should_run

    global microphone_buffer
    global result_buffer

    result_buffer = []
    microphone_buffer = collections.deque(maxlen=BUFFER)

    global result_holder

    result_holder.clear()


    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=handle_audio_chunk)
    print("* microphone on")
    microphone_stop_lock.acquire()
    microphone_running = True
    microphone_stop_lock.release()

    stream.start_stream()

    t = threading.Thread(target=inference_run)
    t.start()

    while stream.is_active():
        time.sleep(0.1)

    print("* microphone off")
    
    stream.stop_stream()
    stream.close()
    
    p.terminate()

class ToggleButton(tk.Button):

    def __preprocess(self, image_path):
        # img = Image.open(image_path)
        # img = img.resize((32, 32))
        # return ImageTk.PhotoImage(img)

        return tk.PhotoImage(file=image_path)

    def __init__(self, parent, image_up, image_down, on_toggle=None, *args, **kwargs):



        self.image_up = self.__preprocess(image_up)
        self.image_down = self.__preprocess(image_down)
        self.on_toggle = on_toggle

        self.__btn_value = False

        tk.Button.__init__(self, parent, image=self.image_up, command=self.toggle, *args, **kwargs)

    def toggle(self):
        self.__btn_value = not self.__btn_value
        if self.__btn_value == False:
            self['image'] = self.image_up
        else:
            self['image'] = self.image_down
        if not self.on_toggle is None:
            self.on_toggle(self.__btn_value)



class FileSelector(tk.Frame):

    def __init__(self, parent, label, on_file_selected, accept_files=(), *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.on_file_selected = on_file_selected
        self.accept_files = accept_files

        self.file_name = tk.StringVar(value="")

        self.label = tk.Label(self, text=label)
        self.entry = tk.Entry(self, textvariable=self.file_name, state='readonly')
        self.button = tk.Button(self, text="Select", command=self.__on_btn_press)

        self.label.pack(side=tk.LEFT, fill=tk.NONE, expand=tk.FALSE)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=tk.TRUE)
        self.button.pack(side=tk.LEFT, fill=tk.NONE, expand=tk.FALSE)

    def __on_btn_press(self):
        f = filedialog.askopenfilename(initialdir="/", title="Open File", filetypes=self.accept_files)
        self.file_name.set(f)
        self.on_file_selected(f)

    def get_file_name(self):
        return self.file_name.get()

class OptionSelector(tk.Frame):

    def __init__(self, parent, label, on_option_selected, options, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.on_option_selected = on_option_selected
        self.options = options
        self.selected = tk.StringVar(value=options[0])

        self.label = tk.Label(self, text=label)
        self.option_menu = tk.OptionMenu(self, self.selected, *options, command=self.__on_option_select)

        self.label.pack(side=tk.LEFT, fill=tk.NONE, expand=tk.FALSE)
        self.option_menu.pack(side=tk.LEFT, fill=tk.X, expand=tk.TRUE)

    def __on_option_select(self, value):
        self.on_option_selected(value)

    def get_option_selected(self):
        return self.selected.get()

class ModelSelector(tk.LabelFrame):

    def __on_model_file_change(self, model_path):
        self.model_path = model_path
        self.model_status_text.set("Model: Not Loaded")
        self.model = None

    def __on_model_type_change(self, model_type):
        self.model_type = model_type

    def __on_load_btn_clicked(self):
        global model

        if not (self.model_path and self.model_type):
            messagebox.showerror("Cannot Load Model", "Cannot Load Model")
            return

        self.model = torch.jit.load(self.model_path, map_location=device)
        model = self.model

        self.model_status_text.set("Model: Loaded")
        

    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, text="Model", *args, **kwargs)
        self.parent = parent

        self.model_path = None
        self.model = None

        self.model_status_text = tk.StringVar(value="Model: Not Loaded")

        self.model_input_file = FileSelector(self, "Model File: ", on_file_selected=lambda x: self.__on_model_file_change(x), accept_files=(("Model Files", "*.pt"),))
        self.model_input_type = OptionSelector(self, "Model Type: ", on_option_selected=lambda x: self.__on_model_type_change(x), options=[MODEL_TYPE_CATEGORICAL, MODEL_TYPE_REGRESSION])
        
        self.model_type = self.model_input_type.get_option_selected()
        
        self.model_btn = tk.Button(self, text="Load", command=self.__on_load_btn_clicked)
        self.model_status = tk.Label(self, textvariable=self.model_status_text)

        self.model_input_file.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        self.model_input_type.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        self.model_btn.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        self.model_status.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)

class MicrophoneController(tk.LabelFrame):

    def __mic_start(self):
        t = threading.Thread(target=listen_microphone)
        t.start()

    def __mic_stop(self):
        global microphone_running
        microphone_stop_lock.acquire()
        microphone_running = False
        microphone_stop_lock.release()

        global result_holder
        result_holder.compute_agg()
    
    def __init__(self, parent, *args, **kwargs):

        tk.LabelFrame.__init__(self, parent, text="Microphone", *args, **kwargs)

        self.parent = parent

        self.mic_toggle_btn = ToggleButton(
            self,
            image_up=get_resource_path("mic_off.png"),
            image_down=get_resource_path("mic_on.png"),
            on_toggle=lambda x: self.__mic_start() if x else self.__mic_stop()
            )

        self.mic_toggle_btn.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)

class ResultBar(tk.Frame):

    def __init__(self, parent, label="", orientation=tk.VERTICAL, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.pbar_val = tk.DoubleVar(value=0.0)
        self.val_var = tk.StringVar(value="0.0")

        self.lbl = tk.Label(self, text=label)
        self.val_lbl = tk.Label(self, textvariable=self.val_var)
        self.pbar = ttk.Progressbar(self, variable=self.pbar_val, length=100, maximum=1.0, orient=orientation)

        self.lbl.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        self.pbar.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        self.val_lbl.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)


    def set_value(self, v):
        self.pbar_val.set(v)
        self.val_var.set(str(v))

class CategoricalResultController(tk.LabelFrame):

    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, text="Categorical Result", *args, **kwargs)
        self.parent = parent

        self.frame_0 = tk.Frame(self)

        self.bars = [
            ResultBar(self.frame_0, "Happy"),
            ResultBar(self.frame_0, "Angry"),
            ResultBar(self.frame_0, "Sad"),
            ResultBar(self.frame_0, "Calm")
        ]

        self.frame_1 = tk.Frame(self)
        self.lbl_final = tk.Label(self.frame_1, text="Final Result")
        self.acc_bars = [
            ResultBar(self.frame_1, "Happy", orientation=tk.HORIZONTAL),
            ResultBar(self.frame_1, "Angry", orientation=tk.HORIZONTAL),
            ResultBar(self.frame_1, "Sad", orientation=tk.HORIZONTAL),
            ResultBar(self.frame_1, "Calm", orientation=tk.HORIZONTAL)
        ]

        self.frame_0.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        for x in self.bars:
            x.pack(side=tk.LEFT, fill=tk.X, expand=tk.TRUE)

        self.frame_1.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        self.lbl_final.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)
        for x in self.acc_bars:
            x.pack(side=tk.LEFT, fill=tk.X, expand=tk.TRUE)

    def set_value(self, result: torch.Tensor):
        global result_buffer
        result_buffer.append(result[0])
        for x in range(0, 4):
            self.bars[x].set_value(round(result[0][x].item(), 2))

    def compute_agg(self):
        global result_buffer
        if len(result_buffer) == 0:
            return
        agg = result_buffer[0]
        for i in range(1, len(result_buffer)):
            agg += result_buffer[i]
        agg = agg / len(result_buffer)
        for x in range(0, 4):
            self.acc_bars[x].set_value(round(agg[x].item(), 2))

    def clear(self):
        for x in range(0, 4):
            self.bars[x].set_value(0.0)
        for x in range(0, 4):
            self.acc_bars[x].set_value(0.0)

class RegressionResultController(tk.LabelFrame):

    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, text="Regression Result", *args, **kwargs)
        self.parent = parent

        f = Figure(figsize=(5,5), dpi=100)

        self.history = collections.deque(maxlen=5)
        
        a = f.add_subplot(111)

        self.figure = f
        self.plt = a

        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        self.history.append([-.5, .5])
        self.history.append([-.5, .5])
        self.history.append([-.5, .5])
        self.history.append([-.5, .5])
        self.history.append([-.5, .5])

        self.draw_plot()

        

    def draw_plot(self):

        self.plt.cla()

        self.plt.axhline(0, color='black')
        self.plt.axvline(0, color='black')

        self.plt.set_xlim([-1, 1])
        self.plt.set_ylim([-1, 1])

        self.plt.set_xlabel("Valence")
        self.plt.set_ylabel("Arousal")

        for (i, x) in enumerate(self.history):
            alpha = np.power(((i+1) / len(self.history)), 2)
            self.plt.scatter([x[0]], [x[1]], c="blue", alpha=alpha)
            self.canvas.draw()

    def set_value(self, result: torch.Tensor):
        
        valence_mean = result[0][0].item()
        arousal_mean = result[0][1].item()

        self.history.append((valence_mean, arousal_mean))
        self.draw_plot()
        # pass
        # global result_buffer
        # result_buffer.append(result[0])
        # for x in range(0, 4):
        #     self.bars[x].set_value(round(result[0][x].item(), 2))

    def compute_agg(self):
        pass
        # global result_buffer
        # if len(result_buffer) == 0:
        #     return
        # agg = result_buffer[0]
        # for i in range(1, len(result_buffer)):
        #     agg += result_buffer[i]
        # agg = agg / len(result_buffer)
        # for x in range(0, 4):
        #     self.acc_bars[x].set_value(round(agg[x].item(), 2))

    def clear(self):
        pass
        # for x in range(0, 4):
        #     self.bars[x].set_value(0.0)
        # for x in range(0, 4):
        #     self.acc_bars[x].set_value(0.0)


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        global model_type

        # self.model_selector = ModelSelector(self)
        self.mic_controller = MicrophoneController(self)

        if model_type == "categorical":
            self.result_holder = CategoricalResultController(self)
        else:
            self.result_holder = RegressionResultController(self)

        global result_holder
        result_holder = self.result_holder

        # self.model_selector.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE, padx=8, pady=4)
        self.mic_controller.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE, padx=8, pady=4)
        self.result_holder.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE, padx=8, pady=4)

def main():
    global model
    global model_type

    n_args = len(sys.argv)
    if n_args < 4:
        print("exec [run_name] [version] [run_id] [type=categorical]")
        print("Example: python main.py 1dconv.cat.a 1 kk7mn5lm [categorical]")
        return

    print("Model: Loading...")

    run_name = sys.argv[1]
    version = sys.argv[2]
    run_id = sys.argv[3]
    model_type = sys.argv[4] if n_args > 4 else "categorical"

    model = get_model(run_name, run_id, version)

    print("Model: Loaded")

    root = tk.Tk()
    MainApplication(root).pack(side=tk.TOP, fill=tk.BOTH, expand=tk.FALSE)
    root.mainloop()

if __name__ == "__main__":
    main()