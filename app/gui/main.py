from tkinter import Entry, StringVar, Tk, Frame, filedialog, Text, Label, LabelFrame, Button, OptionMenu
import tkinter as tk
import pygame
import pyaudio
import threading
import torch

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
BUFFER = RATE * 10

MODEL_TYPE_CATEGORICAL = "Categorical"
MODEL_TYPE_REGRESSION = "Regression"

model = None

microphone_stop_lock = threading.Lock()
microphone_running = False

device = "gpu" if torch.cuda.is_available() else "cpu"

def get_model_load_status_string():
    if model is None:
        return "Status: Not Loaded"
    return "Status: Loaded"

## Microphone
def __bufferSize(frames):
    t = 0
    for x in frames:
        t += len(x)
    return t

def listen_microphone():
    global microphone_running
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    microphone_stop_lock.acquire()
    microphone_running = True
    microphone_stop_lock.release()
    while microphone_running:
        data = stream.read(CHUNK)
        frames.append(data)
        print(__bufferSize(frames))
        if __bufferSize(frames) > BUFFER:
            frames = []
    print("* stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()

def start_microphone_listen():
    t = threading.Thread(target=listen_microphone)
    t.start()

def stop_microphone_listen():
    global microphone_running
    microphone_stop_lock.acquire()
    microphone_running = False
    microphone_stop_lock.release()

def on_closing():
    stop_microphone_listen()
    root.destroy()

# print(model)

root = Tk()
root.title("App")
root.geometry("400x300")

model_input_root = Frame(root, padx=2, pady=4)

model_input_frame = LabelFrame(model_input_root, text='Model', padx=2, pady=4,)

model_input_select_frame = Frame(model_input_frame,)

## file to browse .pt file
model_input_label = Label(model_input_select_frame, text='Model File: ')
model_input_label.pack(fill=tk.NONE, side=tk.LEFT, expand=tk.FALSE)

## open file text field
model_input_text = tk.StringVar()
model_input_text.set('Select')
model_input_file_entry = Entry(model_input_select_frame, textvariable=model_input_text, state=tk.DISABLED)
model_input_file_entry.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.TRUE)

## open file button
def open_model_file():
    global model_input_text
    x = filedialog.askopenfilename(initialdir="/", title="Open Audio File", filetypes=(("pt files", "*.pt"),))
    model_input_text.set(x)

model_input_file_btn = Button(model_input_select_frame, text='Select', command=open_model_file)
model_input_file_btn.pack(fill=tk.X, side=tk.LEFT, expand=tk.FALSE)

model_input_select_frame.pack(fill=tk.BOTH, side=tk.TOP)

model_input_submission_frame = Frame(model_input_frame,)

## model type label
model_input_type_label = Label(model_input_submission_frame, text="Model Type: ")
model_input_type_label.pack(side=tk.LEFT)

## model type select dropdown
model_input_type_variable = tk.StringVar()
model_input_type_variable.set(MODEL_TYPE_CATEGORICAL)
model_input_type_option = OptionMenu(model_input_submission_frame, model_input_type_variable, MODEL_TYPE_CATEGORICAL, MODEL_TYPE_REGRESSION)
model_input_type_option.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.FALSE)

model_input_submission_frame.pack(fill=tk.BOTH, side=tk.TOP)

model_load_status_variable = StringVar()
model_load_status_variable.set(get_model_load_status_string())
model_load_status_label = Label(model_input_frame, textvariable=model_load_status_variable)

def load_model():
    global model
    global device
    global model_input_text
    global model_load_status_variable
    model = torch.jit.load(model_input_text.get(), map_location=device)
    model_load_status_variable.set(get_model_load_status_string())

model_load_btn = Button(model_input_frame, text="Load", command=load_model)

model_load_btn.pack(fill=tk.NONE, side=tk.LEFT, expand=tk.FALSE)
model_load_status_label.pack(fill=tk.NONE, side=tk.RIGHT, expand=tk.FALSE)

model_input_frame.pack(fill=tk.BOTH, side=tk.TOP, expand=tk.FALSE)

model_input_root.pack(fill=tk.BOTH, side=tk.TOP, expand=tk.FALSE)

## AUDIO INPUT

audio_input_root = Frame(root, padx=2, pady=4, )

audio_file_input_frame = LabelFrame(audio_input_root, text="Audio File Input")

def open_audio_file():
    audio_file_path = filedialog.askopenfilename(initialdir="/", title="Open Audio File", filetypes=(("mp3 files", "*.mp3"),))
    

audio_open_btn = Button(audio_file_input_frame, text="Open File", command=)


audio_file_input_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=tk.FALSE)

microphone_input_frame = LabelFrame(audio_input_root, text="Microphone Input")

## microphone toggle button
microphone_toggle_variable = StringVar(value="Turn On")
microphone_toggle_btn = Button(microphone_input_frame, textvariable=microphone_toggle_variable)
microphone_toggle_btn.pack(fill=tk.BOTH)

## microphone status button
microphone_status_variable = StringVar(value="Status: Not Active")
microphone_status_lbl = Label(microphone_input_frame, textvariable=microphone_status_variable)
microphone_status_lbl.pack(fill=tk.BOTH)

microphone_input_frame.pack(fill=tk.BOTH, side=tk.RIGHT, expand=tk.FALSE)

audio_input_root.pack(fill=tk.BOTH, side=tk.TOP, expand=tk.FALSE)

# model_input_frame.grid(column=0, row=0, columnspan=3, rowspan=2)
# model_input_type_option.grid(column=0, row=0)

# # model_input_frame.pack(fill='both', side='top', expand='False')

# frameInput = Frame(root, bg='red')
# frameInput.pack(fill='both', side='top', expand='True')

# frameAudio = Frame(frameInput, bg='purple')
# frameAudio.pack(fill='both', side='left', expand='True')

# frameMic = Frame(frameInput, bg='green')
# frameMic.pack(fill='both', side='right', expand='True')

# frameResult = Frame(root, bg='blue')
# frameResult.pack(fill='both', side='bottom', expand='True')

# ## File Opener
# # root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

# button = Button(root, text="Open File", command=openFile)
# button.pack()

# ## Scrollbar
# frame1 = Frame(root)
# w = Scale(frame1, from_=0, to=3.5, resolution=0.01, showvalue=0, orient=HORIZONTAL)
# w.pack()
# frame1.pack()

# button = Button(root, text="Microphone On", command=start_microphone_listen)
# button.pack()

# button = Button(root, text="Microphone Off", command=stop_microphone_listen)
# button.pack()

# root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()