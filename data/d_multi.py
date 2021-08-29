from os import path

import pylrc
import pylrc.classes

import torchaudio

import numpy as np

from data import *
from data.base import BaseChunkedDataset

class DAudioLyricsDataset(BaseChunkedDataset):

    def __init__(self, meta_file, data_dir, sr=22050, chunk_duration=5, overlap=2.5, temp_folder=None, force_compute=False, audio_extension="mp3"):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute)

        self.sr = sr
        self.audio_dir = path.join(data_dir, "audio")
        self.lyrics_dir = path.join(data_dir, "lyrics")
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

    def get_lyrics(self, info, args):

        if info['lyrics'] != 1:
            return None

        frame = args

        lyrics_file = path.join(self.lyrics_dir, "{}.lrc".format(info['song_id']))
        lrc_string = ''
        with open(lyrics_file, mode='r') as f:
            lrc_string = ''.join(f.readlines())

        song_start_time = info[SONG_START_TIME]
        start_time = (song_start_time - 3) + (self.overlap * frame)
        end_time = (song_start_time - 3) + (self.overlap * frame) + self.chunk_duration

        ret = ""
        try:
            subs = pylrc.parse(lrc_string)
            subs = subs.getLyricsBetween(start_time, end_time)
            t = " ".join(list(map(lambda x: x.text, subs)))
            ret = t
        except Exception as e:
            print(e)
            print(lyrics_file)
            print(start_time, end_time)

        return ret

    def get_label(self, info, args):
        frame = args
        annotation_rate = 0.5
        start_i = int((self.overlap * frame) / annotation_rate)
        end_i = start_i + int((self.chunk_duration) / annotation_rate)
        x = []
        for l in [DYNAMIC_VALENCE_MEAN, DYNAMIC_AROUSAL_MEAN, DYNAMIC_VALENCE_STD, DYNAMIC_AROUSAL_STD]:
            y = np.array(info[l])
            x.append(y)
        x = np.array(x)
        x = x[:, start_i:end_i]
        x = np.mean(x, axis=1)
        y = torch.tensor(x, dtype=torch.float)
        return y

    def get_features(self, info, args):
        audio_x, sr = self.get_audio(info, args)
        audio_x = preprocess_audio(self.frame_count, audio_x, sr, self.sr)
        
        lyrics_x = self.get_lyrics(info, args)

        return (audio_x, lyrics_x)