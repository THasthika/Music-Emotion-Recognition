from os import path

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
                 duration=30):
        super().__init__(meta_file, sample_rate, duration)
        self.audio_dir = audio_dir
        self.label_type = label_type

    def get_labels(self):
        if self.label_type == "categorical":
            return self.meta['quadrants']
        raise NotImplementedError

    def get_label(self, index):
        item = self.meta.iloc[index]
        if self.label_type == "categorical":
            label = item['quadrant']
            return label
        raise NotImplementedError

    def get_audio(self, index):
        item = self.meta.iloc[index]
        audio_file = path.join(self.audio_dir, "{}.mp3".format(item['song_id']))
        x, sr = torchaudio.load(audio_file)
        return (x, sr)