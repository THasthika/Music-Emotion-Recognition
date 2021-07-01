import torch
from torch.autograd.grad_mode import F
import torchaudio

from torch.utils.data import Dataset

import random
import string
import os
from os import path
import pickle
import numpy as np
import pandas as pd

import librosa

SONG_ID = 'song_id'
START_TIME = 'start_time'
END_TIME = 'end_time'
CLIP_START_TIME = 'clip_start_time'
DURATION = 'duration'

QUADRANT = 'quadrant'
STATIC_VALENCE_MEAN = 'static_valence_mean'
STATIC_VALENCE_STD = 'static_valence_std'
STATIC_AROUSAL_MEAN = 'static_arousal_mean'
STATIC_AROUSAL_STD = 'static_arousal_std'
DYNAMIC_VALENCE_MEAN = 'dynamic_valence_mean'
DYNAMIC_VALENCE_STD = 'dynamic_valence_std'
DYNAMIC_AROUSAL_MEAN = 'dynamic_arousal_mean'
DYNAMIC_AROUSAL_STD = 'dynamic_arousal_std'

def preprocess_audio(frame_count, audio, sr, ret_sr):
    x = torch.mean(audio, 0, True)
    out = torch.zeros(1, frame_count)
    effects = [
        ["rate", f"{ret_sr}"]
    ]
    x, sr2 = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
    if frame_count >= x.shape[1]:
        out[:, :x.shape[1]] = x
    else:
        out[:, :] = x[:, :frame_count]
    # out = torch.squeeze(out)
    # out = torch.unsqueeze(out, dim=1)
    return out


class BaseDataset(Dataset):

    def __get_temp_folder(self, length=8):
        fname = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=length))
        return path.join("/tmp/", fname)

    def __check_cache_and_get_features(self, info, args):
        key = self.get_key(info, args)
        if self.cache_in_memory and key in self.cache:
            return self.cache[key]

        fkey = path.join(self.temp_folder, "{}.pkl".format(key))
        if (not self.force_compute) and path.exists(fkey):
            try:
                X = pickle.load(open(fkey, mode="rb"))
                return X
            except:
                print("Warning: failed to load pickle file. getting features... {}".format(fkey))
        X = self.get_features(info, args)
        pickle.dump(X, open(fkey, mode="wb"))
        if self.cache_in_memory:
            self.cache[key] = X

        return X

    def __get_meta(self, meta_file):
        meta_ext = path.splitext(meta_file)[1]
        if meta_ext == ".json":
            return pd.read_json(meta_file)
        elif meta_ext == ".csv":
            return pd.read_csv(meta_file)
        else:
            raise Exception("Unknown File Extension {}".format(meta_ext))

    def __init__(self, meta_file, temp_folder=None, force_compute=False, cache_in_memory=False):

        self.cache_in_memory = cache_in_memory
        self.cache = {}

        self.force_compute = force_compute
        self.meta = self.__get_meta(meta_file)
        if force_compute == True:
            print("Warning: Using Force Compute")

        if temp_folder is None:
            temp_folder = self.__get_temp_folder()
        if not path.exists(temp_folder):
            os.mkdir(temp_folder)
        self.temp_folder = temp_folder

        self.count = len(self.meta)

    def get_key(sekf, info, args):
        k = "{}".format(info[SONG_ID])
        if not args is None:
            k += "-{}".format(args)
        return k

    def get_features(self, info, args):
        raise NotImplementedError()

    def get_label(self, info, args):
        return info[QUADRANT]

    def get_info(self, index):
        return (self.meta.iloc[index], None)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        (info, args) = self.get_info(index)
        X = self.__check_cache_and_get_features(info, args)
        # X = self.get_features(info, args)
        y = self.get_label(info, args)
        return (X, y)


class BaseChunkedDataset(BaseDataset):

    def __calculate_frames(self, meta, chunk_duration, overlap):
        frames = []
        row_i = 0
        for (i, row) in meta.iterrows():
            start_time = row[START_TIME]
            end_time = row[END_TIME]
            duration = end_time - start_time
            n_frames = int(np.floor((duration - 1) /
                           (chunk_duration - overlap)))
            if n_frames == 0 and chunk_duration <= duration:
                frames.append((row_i, 0))
                continue
            for j in range(n_frames):
                frames.append((row_i, j))
            row_i += 1
        return frames

    def __init__(self, meta_file, chunk_duration=5, overlap=2.5, temp_folder=None, force_compute=False, cache_in_memory=False):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute, cache_in_memory=cache_in_memory)

        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.frames = self.__calculate_frames(
            self.meta, self.chunk_duration, self.overlap)
        self.count = len(self.frames)

    def get_info(self, index):
        (meta_index, frame) = self.frames[index]
        info = self.meta.iloc[meta_index]
        return (info, frame)

class AudioChunkedDataset(BaseChunkedDataset):

    def __init__(self, meta_file, data_dir, sr=22050, chunk_duration=5, overlap=2.5, temp_folder=None, force_compute=False, cache_in_memory=False, audio_extension="mp3"):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute, cache_in_memory=cache_in_memory)

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

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        x = preprocess_audio(self.frame_count, x, sr, self.sr)
        return x

class ModelDataset(AudioChunkedDataset):

    def __init__(self, meta_file, data_dir, sr=22050, chunk_duration=5, overlap=2.5, temp_folder=None, force_compute=False, cache_in_memory=False, audio_extension="mp3", frame_size=1024, hop_size=512, n_fft=1024, n_mels=128, n_mfcc=20, n_chroma=12, n_spectral_contrast_bands=6):
        super().__init__(meta_file, data_dir, sr=sr, chunk_duration=chunk_duration, overlap=overlap, temp_folder=temp_folder, force_compute=force_compute, cache_in_memory=cache_in_memory, audio_extension=audio_extension)

        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_spectral_contrast_bands = n_spectral_contrast_bands

    def __compute_features(self, audio, frame_size=1024, hop_size=512, sample_rate=22050, n_fft=1024, n_mels=128, n_mfcc=20, n_chroma=12, n_spectral_contrast_bands=6):

        sr = sample_rate

        # calculate stft (n_fft//2 + 1) - features
        spec = np.abs(librosa.spectrum.stft(
            audio, win_length=frame_size, n_fft=n_fft, hop_length=hop_size, window="hann"))
        power_spec = np.power(spec, 2)

        # calculate mel spectrogram (n_mels) - features
        mel_spec = librosa.feature.melspectrogram(
            S=power_spec, sr=sr, n_fft=frame_size, n_mels=n_mels)

        # calculate mfccs (n_mfcc) - features
        mfccs = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

        # calculate chroma (n_chroma) - features
        chroma = librosa.feature.chroma_stft(S=spec, sr=sr, n_chroma=n_chroma)

        # calculate the tonnetz (perfect 5th, minor and major 3rd all in 2 d) - 6
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr, chroma=chroma)

        # calculate spectral contrast - (n_spectral_contrast_bands + 1)
        spectral_contrast = librosa.feature.spectral_contrast(
            S=spec, sr=sr, n_bands=n_spectral_contrast_bands)

        # initialize array
        offset = 0
        feature_count = 6
        frame_count = spec.shape[1]
        arr = np.zeros((feature_count, frame_count))

        # calculate tempo - 1
        onset_env = librosa.onset.onset_strength(audio, sr=sr)
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=None)
        arr[offset:offset+1] = tempo
        offset += 1

        # calculate spectral centroid - 1
        spectral_centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)
        arr[offset:offset+1] = spectral_centroid
        offset += 1

        # calculate spectral bandwidth - 1
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sr)
        arr[offset:offset+1] = spectral_bandwidth
        offset += 1

        # calculate spectral flatness - 1
        spectral_flatness = librosa.feature.spectral_flatness(S=spec)
        arr[offset:offset+1] = spectral_flatness
        offset += 1

        # calculate spectral rolloff - 1
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=spec, sr=sr, roll_percent=0.85)
        arr[offset:offset+1] = spectral_rolloff
        offset += 1

        # calculate zero crossing rate - 1
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=frame_size, hop_length=hop_size)
        arr[offset:offset+1] = zcr

        ret = {
            'spec': torch.tensor(spec, dtype=torch.float32),
            'mel_spec': torch.tensor(mel_spec, dtype=torch.float32),
            'mfccs': torch.tensor(mfccs, dtype=torch.float32),
            'chroma': torch.tensor(chroma, dtype=torch.float32),
            'tonnetz': torch.tensor(tonnetz, dtype=torch.float32),
            'spectral_contrast': torch.tensor(spectral_contrast, dtype=torch.float32)
        }

        return {
            **ret,
            'spectral_aggregate': torch.tensor(arr, dtype=torch.float32)
        }

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        raw_audio = preprocess_audio(self.frame_count, x, sr, self.sr)

        computed = self.__compute_features(np.array(torch.squeeze(raw_audio)),
                                           frame_size=self.frame_size,
                                           hop_size=self.hop_size,
                                           n_fft=self.n_fft,
                                           n_mels=self.n_mels,
                                           n_mfcc=self.n_mfcc,
                                           n_chroma=self.n_chroma,
                                           n_spectral_contrast_bands=self.n_spectral_contrast_bands
                                           )

        x = {
            'audio': raw_audio,
            **computed
        }

        return x

    def get_label(self, info, args):
        y = info[['static_valence_mean', 'static_arousal_mean',
                  'static_valence_std', 'static_arousal_std']].to_numpy()
        y = torch.tensor(list(y), dtype=torch.float)
        return y