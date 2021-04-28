from os import path

import torch
from torch.utils.data import Dataset

import torchaudio

import pandas as pd
import numpy as np

class BaseAudioDataset(Dataset):

    def __init__(self, meta_file, sample_rate=22050, chunk_duration=5, overlap=2.5):

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.frame_count = self.sample_rate * self.chunk_duration

        meta_ext = path.splitext(meta_file)[1]
        if meta_ext == ".json":
            self.meta = pd.read_json(meta_file)
        elif meta_ext == ".csv":
            self.meta = pd.read_csv(meta_file)
        else:
            raise Exception("Unknown File Extension {}".format(meta_ext))

        self.meta = self.meta

        self.count = self.__calculate_count(self.meta, self.chunk_duration, self.overlap)

    def __calculate_count(self, meta, chunk_duration, overlap):
        self.frames = []
        row_i = 0
        for (i, row) in meta.iterrows():
            duration = row['duration']
            n_frames = int(np.floor((duration - 1) / (chunk_duration - overlap)))
            if n_frames == 0 and chunk_duration <= duration:
                self.frames.append((row_i, 0))
                continue
            for j in range(n_frames):
                self.frames.append((row_i, j))
            row_i += 1
        return len(self.frames)


    def __len__(self):
        return self.count

    def get_labels(self):
        raise NotImplementedError

    def get_label(self, info, frame):
        raise NotImplementedError

    def get_audio(self, info, frame):
        raise NotImplementedError

    def __getitem__(self, index):

        (meta_index, frame) = self.frames[index]
        info = self.meta.iloc[meta_index]

        x, sr = self.get_audio(info, frame)
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

        y = self.get_label(info, frame)
        
        return (out, y)