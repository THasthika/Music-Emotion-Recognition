from tkinter import *
import pygame
import pyaudio
import torch

device = "gpu" if torch.cuda.is_available() else "cpu"

model = torch.jit.load("./serving/models/n1dconv/cat/a/model-mer-taffc.pt", map_location=device)

print(model)

root = Tk()
root.title("App")

root.geometry("300x200")

x = torch.rand((10, 1, 22050*5))
print(model(x))

root.mainloop()