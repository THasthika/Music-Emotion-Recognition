from os import path

import torchaudio

from .base import BaseAudioDataset

class MERTaffcDataset(BaseAudioDataset):

    """
    audio_dir="./drive/MyDrive/Research/Datasets/Raw/mer-taffc/audio",
    meta_file="./drive/MyDrive/Research/Datasets/Splits/mer-taffc-train-70-test-30-seed-42/train.json",
    """

    def __init__(self,
                 audio_dir,
                 meta_file,
                 sample_rate=22050,
                 duration=30):
        super().__init__(meta_file, sample_rate, duration)
        self.audio_dir = audio_dir

    def get_labels(self):
        return self.meta[:, -1:]

    def get_label(self, index):
        item = self.meta.iloc[index]
        label = item['quadrant']
        return label

    def get_audio(self, index):
        item = self.meta.iloc[index]
        audio_file = path.join(self.audio_dir, "{}.mp3".format(item['song_id']))
        x, sr = torchaudio.load(audio_file)
        return (x, sr)