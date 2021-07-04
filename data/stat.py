from os import path

import torchaudio

from data import *
from data.base import BaseChunkedDataset

class StatAudioDataset(BaseChunkedDataset):

    def __init__(self, meta_file, data_dir, sr=22050, chunk_duration=5, overlap=2.5, temp_folder=None, force_compute=False, audio_extension="mp3"):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute)

        self.sr = sr
        self.data_dir = data_dir
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.audio_extension = audio_extension
        self.frame_count = int(self.sr * self.chunk_duration)

    def get_audio(self, info, args):
        audio_file = path.join(self.data_dir, "{}.{}".format(
            info[SONG_ID], self.audio_extension))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate

        frame = args
        offset = int(sr * ((self.overlap * frame) + info[START_TIME]))
        frames = int(sr * self.chunk_duration)

        x, sr = torchaudio.load(
            audio_file, frame_offset=offset, num_frames=frames)
        return (x, sr)

    def get_label(self, info, args):
        y = info[['static_valence_mean', 'static_arousal_mean',
                  'static_valence_std', 'static_arousal_std']].to_numpy()
        y = torch.tensor(list(y), dtype=torch.float)
        return y

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        x = preprocess_audio(self.frame_count, x, sr, self.sr)
        return x

