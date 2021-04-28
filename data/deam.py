from os import path

import numpy as np

import torch
import torchaudio

from .base import BaseAudioDataset

class DeamDataset(BaseAudioDataset):

    """
    label_type = (categorical|static|dynamic)
    """

    def __init__(self,
                 audio_dir,
                 meta_file,
                 label_type="categorical",
                 sample_rate=22050,
                 chunk_duration=5,
                 overlap=2.5):
        super().__init__(meta_file, sample_rate, chunk_duration, overlap)
        self.audio_dir = audio_dir
        self.label_type = label_type

    def get_labels(self):
        if self.label_type == "categorical":
            return self.meta['quadrants']
        if self.label_type == "static":
            return self.meta[['static_valence_mean', 'static_valence_std', 'static_arousal_mean', 'static_arousal_std']]
        if self.label_type == "dynamic":
            return self.meta[['dynamic_valence_mean', 'dynamic_valence_std', 'dynamic_arousal_mean', 'dynamic_arousal_std']]
        raise NameError

    def get_label(self, info, frame):
        if self.label_type == "categorical":
            label = info['quadrant']
            return label
        if self.label_type == "static":
            ret_cols = ['static_valence_mean', 'static_valence_std', 'static_arousal_mean', 'static_arousal_std']
            return torch.tensor(info[ret_cols].astype(np.float32))
        # if self.label_type == "dynamic":
        #     ret_cols = ['dynamic_valence_mean', 'dynamic_valence_std', 'dynamic_arousal_mean', 'dynamic_arousal_std']
        #     ret = []
        #     x = info[ret_cols]
        #     for i in range(len(x[0])):
        #         t = []
        #         for j in range(len(ret_cols)):
        #             t.append(x[j][i])
        #         ret.append(t)
        #     return torch.tensor(ret)
        raise NameError

    def get_audio(self, info, frame):
        audio_file = path.join(self.audio_dir, "{}.mp3".format(info['song_id']))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate

        offset = int(sr * self.overlap * frame)
        frames = int(sr * self.chunk_duration)

        x, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=frames)
        return (x, sr)