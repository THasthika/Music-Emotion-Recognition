from os import path

import torchaudio

from data import *
from data.base import BaseChunkedDataset

import numpy as np
import librosa

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
        y = info[[STATIC_VALENCE_MEAN, STATIC_AROUSAL_MEAN,
                  STATIC_VALENCE_STD, STATIC_AROUSAL_STD]].to_numpy()
        y = torch.tensor(list(y), dtype=torch.float)
        return y

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        x = preprocess_audio(self.frame_count, x, sr, self.sr)
        return x

class StatAudioExtractedDataset(BaseChunkedDataset):

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

    def get_stft(self, audio, n_fft=1024):
        return librosa.stft(audio, n_fft=n_fft)

    def get_mel_spec(self, stft):
        D = np.abs(stft)**2
        return librosa.feature.melspectrogram(S=D, sr=self.sr)

    def get_mfcc(self, mel_spec):
        return librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))

    def get_label(self, info, args):
        y = info[[STATIC_VALENCE_MEAN, STATIC_AROUSAL_MEAN,
                  STATIC_VALENCE_STD, STATIC_AROUSAL_STD]].to_numpy()
        y = torch.tensor(list(y), dtype=torch.float)
        return y

    def get_features(self, info, args):
        audio_x, sr = self.get_audio(info, args)
        audio_x = preprocess_audio(self.frame_count, audio_x, sr, self.sr)

        ## stft
        audio_t = torch.squeeze(audio_x, 0).numpy()
        stft_x = torch.tensor(self.get_stft(audio_t, n_fft=1024), dtype=torch.float)

        ## mel_spec
        mel_spec_x = torch.tensor(self.get_mel_spec(stft_x), dtype=torch.float)

        ## mfcc
        mfcc_x = torch.tensor(self.get_mfcc(mel_spec_x), dtype=torch.float)

        return (audio_x, stft_x, mel_spec_x, mfcc_x)

