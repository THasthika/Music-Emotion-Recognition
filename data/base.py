from os import path

import torch
from torch.utils.data import Dataset

import torchaudio

import pandas as pd

class BaseAudioDataset(Dataset):

    def __init__(self, meta_file, sample_rate=22050, duration=30):

        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_count = self.sample_rate * self.duration

        meta_ext = path.splitext(meta_file)[1]
        if meta_ext == ".json":
            self.meta = pd.read_json(meta_file)
        elif meta_ext == ".csv":
            self.meta = pd.read_csv(meta_file)
        else:
            raise Exception("Unknown File Extension {}".format(meta_ext))

        self.count = len(self.meta)
        self.meta = self.meta

    def __len__(self):
        return self.count

    def get_labels(self):
        raise NotImplementedError

    def get_label(self, index):
        raise NotImplementedError

    def get_audio(self, index):
        raise NotImplementedError

    def __getitem__(self, index):

        x, sr = self.get_audio(index)
        x = torch.mean(x, 0, True)
        out = torch.zeros(1, self.frame_count)
        
        effects = [
          ["rate", f"{self.sample_rate}"]
        ]
        
        x, sr2 = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
        
        if self.frame_count >= x.shape[1]:
            out[:, :x.shape[1]] = x
        else:
            out[:, :] = x[:, :self.frame_count]

        # out = torch.squeeze(out)
        # out = torch.unsqueeze(out, dim=1)

        y = self.get_label(index)
        
        return (out, y)