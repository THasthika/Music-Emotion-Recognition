from os import path

import numpy as np

import torch
import torchaudio

from data.base import BaseAudioDataset

class FSD50K(BaseAudioDataset):

    def __init__(self,
                 audio_dir,
                 meta_file,
                 label_type="categorical",
                 sample_rate=22050,
                 chunk_duration=5,
                 overlap=2.5):
        super().__init__(meta_file, sample_rate, chunk_duration, overlap)
        self.audio_dir = audio_dir

    def __len__(self):
        return self.count

    def get_labels(self):
        return self.meta['label']

    def get_label(self, info, frame):
        return info['label']

    def get_audio(self, info, frame):
        audio_file = path.join(self.audio_dir, "{}".format(info['name']))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate

        offset = int(sr * self.overlap * frame)
        frames = int(sr * self.chunk_duration)

        x, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=frames)
        return (x, sr)