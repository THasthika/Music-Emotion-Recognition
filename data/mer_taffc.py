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
                 label_type="categorical",
                 sample_rate=22050,
                 chunk_duration=5,
                 overlap=2.5):
        super().__init__(meta_file, sample_rate, chunk_duration, overlap)
        self.audio_dir = audio_dir

    def get_labels(self):
        return self.meta['quadrant']

    def get_label(self, info, frame):
        label = info['quadrant']
        return label

    def get_audio(self, info, frame):
        audio_file = path.join(self.audio_dir, "{}.mp3".format(info['song_id']))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate

        offset = int(sr * self.overlap * frame)
        frames = int(sr * self.chunk_duration)

        x, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=frames)
        return (x, sr)