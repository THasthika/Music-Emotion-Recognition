from tkinter import *

import torch

device = "gpu" if torch.cuda.is_available() else "cpu"

model = torch.jit.load("./serving/models/1dconv/cat/a/model-mer-taffc.pt", map_location=device)

print(model)

root = Tk()
root.title("App")

root.geometry("300x200")

root.mainloop()