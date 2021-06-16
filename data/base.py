from os import path
import os
import sys

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import librosa
import pickle
import random
import string

SONG_ID = 'song_id'
START_TIME = 'start_time'
END_TIME = 'end_time'
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
        fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        return path.join("/tmp/", fname)

    # def __check_cache_and_get_features(self, info, args):
    #     key = self.get_key(info, args)
    #     fkey = path.join(self.temp_folder, "{}.pkl".format(key))
    #     if (not self.force_compute) and path.exists(fkey):
    #         X = pickle.load(open(fkey, mode="rb"))
    #         return X
    #     X = self.get_features(info, args)
    #     pickle.dump(X, open(fkey, mode="wb"))
    #     return X

    def __get_meta(self, meta_file):
        meta_ext = path.splitext(meta_file)[1]
        if meta_ext == ".json":
            return pd.read_json(meta_file)
        elif meta_ext == ".csv":
            return pd.read_csv(meta_file)
        else:
            raise Exception("Unknown File Extension {}".format(meta_ext))

    def __init__(self, meta_file, temp_folder=None, force_compute=False):

        self.force_compute = force_compute
        self.meta = self.__get_meta(meta_file)

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
        # X = self.__check_cache_and_get_features(info, args)
        X = self.get_features(info, args)
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
            n_frames = int(np.floor((duration - 1) / (chunk_duration - overlap)))
            if n_frames == 0 and chunk_duration <= duration:
                frames.append((row_i, 0))
                continue
            for j in range(n_frames):
                frames.append((row_i, j))
            row_i += 1
        return frames

    def __init__(self, meta_file, chunk_duration=10, overlap=5, temp_folder=None, force_compute=False):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute)

        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.frames = self.__calculate_frames(
            self.meta, self.chunk_duration, self.overlap)
        self.count = len(self.frames)

    def get_info(self, index):
        (meta_index, frame) = self.frames[index]
        info = self.meta.iloc[meta_index]
        return (info, frame)


class BaseAudioOnlyDataset(BaseDataset):

    def __init__(self, meta_file, data_dir, sr=22050, duration=5, temp_folder=None, force_compute=False, audio_extension="mp3"):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute)

        self.sr = sr
        self.duration = duration
        self.data_dir = data_dir
        self.audio_extension = audio_extension
        self.frame_count = int(self.sr * self.duration)

    def get_audio(self, info, args):
        audio_file = path.join(self.data_dir, "{}.{}".format(info[SONG_ID], self.audio_extension))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate
        frames = int(sr * self.duration)
        x, sr = torchaudio.load(audio_file, num_frames=frames)
        return (x, sr)

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        x = preprocess_audio(self.frame_count, x, sr, self.sr)
        return x


class BaseAudioOnlyChunkedDataset(BaseChunkedDataset):

    def __init__(self, meta_file, data_dir, sr=22050, chunk_duration=10, overlap=5, temp_folder=None, force_compute=False, audio_extension="mp3"):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute)

        self.sr = sr
        self.data_dir = data_dir
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.audio_extension = audio_extension
        self.frame_count = int(self.sr * self.chunk_duration)

    def get_audio(self, info, args):
        audio_file = path.join(self.data_dir, "{}.{}".format(info[SONG_ID], self.audio_extension))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate

        frame = args
        offset = int(sr * ((self.overlap * frame) + info[START_TIME]))
        frames = int(sr * self.chunk_duration)

        x, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=frames)
        return (x, sr)

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        x = preprocess_audio(self.frame_count, x, sr, self.sr)
        return x

class AudioOnlyStaticQuadrantAndAVValues(BaseAudioOnlyChunkedDataset):

    def __init__(self, meta_file, data_dir, sr=22050, chunk_duration=10, overlap=5, temp_folder=None, force_compute=False, audio_extension="mp3"):
        super().__init__(meta_file, data_dir, sr=sr, chunk_duration=chunk_duration, overlap=overlap, temp_folder=temp_folder, force_compute=force_compute, audio_extension=audio_extension)

    def get_label(self, info, args):
        y = info[['quadrant', 'static_valence_mean', 'static_valence_std', 'static_arousal_mean', 'static_arousal_std']].to_numpy()
        y0 = y[0]
        y1 = torch.tensor(list(y[1:]), dtype=torch.float)
        return (y0, y1)

# class AudioDataset(BaseDataset):

#     def __init__(self, meta_file, temp_folder=None):
#         super().__init__(meta_file, temp_folder=temp_folder)

# class BaseAudioDataset(BaseDataset):

#     def __init__(self, meta_file, sample_rate=22050, max_duration=5):
#         super().__init__(meta_file)

#         self.sample_rate = sample_rate
#         self.max_duration = max_duration
#         self.frame_count = int(self.sample_rate * max_duration)

#         self.count = len(self.meta)

#     def __len__(self):
#         return self.count

#     def get_audio(self, info, args):
#         raise NotImplementedError()

#     def preprocess_audio(self, audio, sr):
        # x = torch.mean(audio, 0, True)
        # out = torch.zeros(1, self.frame_count)

        # effects = [
        #   ["rate", f"{self.sample_rate}"]
        # ]

        # x, sr2 = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)

        # if self.frame_count >= x.shape[1]:
        #     out[:, :x.shape[1]] = x
        # else:
        #     out[:, :] = x[:, :self.frame_count]

        # # out = torch.squeeze(out)
        # # out = torch.unsqueeze(out, dim=1)

        # return out

#     def get_features(self, info, args):
#         x, sr = self.get_audio(info, args)
#         out = self.preprocess_audio(x, sr)
#         return out

#     def get_info(self, index):
#         return (self.meta.iloc[index], None)

# class BaseChunkedAudioOnlyDataset(BaseAudioDataset):

#     def __init__(self, meta_file, sample_rate=22050, chunk_duration=5, overlap=2.5):
#         super().__init__(meta_file, sample_rate, 1)

#         self.chunk_duration = chunk_duration
#         self.overlap = overlap
#         self.frame_count = int(self.sample_rate * self.chunk_duration)

#         self.count = self.__calculate_count(self.meta, self.chunk_duration, self.overlap)

#     def __calculate_count(self, meta, chunk_duration, overlap):
#         self.frames = []
#         row_i = 0
#         for (i, row) in meta.iterrows():
#             duration = row['duration']
#             n_frames = int(np.floor((duration - 1) / (chunk_duration - overlap)))
#             if n_frames == 0 and chunk_duration <= duration:
#                 self.frames.append((row_i, 0))
#                 continue
#             for j in range(n_frames):
#                 self.frames.append((row_i, j))
#             row_i += 1
#         return len(self.frames)

#     def __len__(self):
#         return self.count

#     def get_info(self, index):
#         (meta_index, frame) = self.frames[index]
#         info = self.meta.iloc[meta_index]
#         return (info, frame)

# class GenericStaticAudioOnlyDataset(BaseAudioDataset):

#     def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, max_duration=30):
#         super().__init__(meta_file, sample_rate, max_duration)

#         self.data_dir = data_dir
#         self.audio_index = audio_index
#         self.audio_extension = audio_extension
#         self.label_index = label_index

#     def get_label(self, info, args):
#         if type(self.label_index) is list:
#             return info[self.label_index].to_numpy()
#         return info[self.label_index]

#     def get_audio(self, info, args):
#         audio_file = path.join(self.data_dir, "{}.{}".format(info[self.audio_index], self.audio_extension))
#         meta_data = torchaudio.info(audio_file)
#         sr = meta_data.sample_rate
#         frames = int(sr * self.max_duration)
#         x, sr = torchaudio.load(audio_file, num_frames=frames)
#         return (x, sr)

# class GenericStaticChunkedAudioOnlyDataset(BaseChunkedAudioOnlyDataset):

#     def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, chunk_duration=5, overlap=2.5):
#         super().__init__(meta_file, sample_rate, chunk_duration, overlap)

#         self.data_dir = data_dir
#         self.audio_index = audio_index
#         self.audio_extension = audio_extension
#         self.label_index = label_index

#     def get_label(self, info, args):
#         if type(self.label_index) is list:
#             return info[self.label_index].to_numpy()
#         return info[self.label_index]

#     def get_audio(self, info, args):
#         audio_file = path.join(self.data_dir, "{}.mp3".format(info['song_id']))
#         meta_data = torchaudio.info(audio_file)
#         sr = meta_data.sample_rate

#         frame = args
#         offset = int(sr * self.overlap * frame)
#         frames = int(sr * self.chunk_duration)

#         x, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=frames)
#         return (x, sr)

# class GenericStaticAudioFeatureOnlyDataset(BaseAudioDataset):

#     def __init__(self, meta_file, data_dir, audio_index='song_id', label_index='quadrant'):
#         super().__init__(meta_file)
#         self.data_dir = data_dir
#         self.audio_index = audio_index
#         self.label_index = label_index
#         self.count = len(self.meta)

#     def __len__(self):
#         return self.count

#     def get_info(self, index):
#         info = self.meta.iloc[index]
#         return (info, None)

#     def __stack_matching_keys(self, kset, d):
#         kset_out = np.array(d[kset[0]], dtype='float32')
#         for k in kset:
#             if k == kset[0]:
#                 continue
#             kset_out = np.hstack((kset_out, d[k]))
#         return kset_out

#     def __pad_array(self, arr, size):
#         out = np.zeros((size, arr.shape[1]), dtype='float32')
#         if size >= arr.shape[0]:
#             out[:arr.shape[0], :] = arr
#         else:
#             out[:, :] = arr[:size, :]
#         return out

#     def get_features(self, info, args):
#         name = info[self.audio_index]
#         fname = "{}.npy".format(name)
#         fpath = path.join(self.data_dir, fname)
#         d = np.load(fpath, allow_pickle=True)
#         d = d[()]

#         kset_1 = [
#             'mfcc',
#             'mfcc_bands',
#             'mfcc_bands_log',
#             'zero_crossing_rate',
#             'spectral_centroid',
#         ]

#         kset_2 = [
#             'chromagram'
#         ]

#         kset_3 = [
#             'dancability'
#         ]

#         kset1_out = self.__stack_matching_keys(kset_1, d)
#         kset2_out = self.__stack_matching_keys(kset_2, d)
#         kset3_out = self.__stack_matching_keys(kset_3, d)

#         kset1_out = self.__pad_array(kset1_out, 1294)
#         kset2_out = self.__pad_array(kset2_out, 40)
#         kset3_out = np.reshape(kset3_out, (1, 1))

#         return (kset1_out, kset2_out, kset3_out)

#     def get_label(self, info, args):
#         if type(self.label_index) is list:
#             return info[self.label_index].to_numpy()
#         return info[self.label_index]

# class GenericStaticHybridAudioOnlyDataset(GenericStaticAudioOnlyDataset):

#     def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, max_duration=30):
#         super().__init__(meta_file, path.join(data_dir, "audio"), audio_index, audio_extension, label_index, sample_rate, max_duration)

#         self.feature_dir = path.join(data_dir, "features")

#     def __stack_matching_keys(self, kset, d):
#         kset_out = np.array(d[kset[0]], dtype='float32')
#         for k in kset:
#             if k == kset[0]:
#                 continue
#             kset_out = np.hstack((kset_out, d[k]))
#         return kset_out

#     def __pad_array(self, arr, size):
#         out = np.zeros((size, arr.shape[1]), dtype='float32')
#         if size >= arr.shape[0]:
#             out[:arr.shape[0], :] = arr
#         else:
#             out[:, :] = arr[:size, :]
#         return out

#     def get_extracted_features(self, info, args):
#         name = info[self.audio_index]
#         fname = "{}.npy".format(name)
#         fpath = path.join(self.feature_dir, fname)
#         d = np.load(fpath, allow_pickle=True)
#         d = d[()]

#         kset_1 = [
#             'mfcc',
#             'mfcc_bands',
#             'mfcc_bands_log',
#             'zero_crossing_rate',
#             'spectral_centroid',
#         ]

#         kset_2 = [
#             'chromagram'
#         ]

#         kset_3 = [
#             'dancability'
#         ]

#         kset1_out = self.__stack_matching_keys(kset_1, d)
#         kset2_out = self.__stack_matching_keys(kset_2, d)
#         kset3_out = self.__stack_matching_keys(kset_3, d)

#         kset1_out = self.__pad_array(kset1_out, 1294)
#         kset2_out = self.__pad_array(kset2_out, 40)
#         kset3_out = np.reshape(kset3_out, (1, 1))

#         return (kset1_out, kset2_out, kset3_out)

#     def get_features(self, info, args):

#         x, sr = self.get_audio(info, args)
#         audio_out = self.preprocess_audio(x, sr)

#         features_out = self.get_extracted_features(info, args)

#         out = (
#             audio_out,
#             *features_out
#         )

#         return out

# class GenericStaticChunkedAudioOnlyFeatureExtractionDataset(GenericStaticChunkedAudioOnlyDataset):

#     def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, chunk_duration=5, overlap=2.5, extractor_config={}, temp_folder="/tmp/"):
#         super().__init__(
#             meta_file=meta_file,
#             data_dir=data_dir,
#             audio_index=audio_index,
#             audio_extension=audio_extension,
#             label_index=label_index,
#             sample_rate=sample_rate,
#             chunk_duration=chunk_duration,
#             overlap=overlap)

#         self.extractor_config = extractor_config

#         self.labels = np.array(
#             list(map(lambda x: self.meta.iloc[x[0]][self.label_index], self.frames))
#         )

#         self.temp_folder = temp_folder

#     def get_labels(self):
#         return self.labels

#     def __compute_features(self, audio, duration, frame_size=2048, hop_size=512, sample_rate=22050, n_fft=2048, n_mels=128, n_mfcc=20, n_chroma=12, spectral_contrast_bands=6):

#         sr = sample_rate

#         stft_feature_count = (1 + n_fft//2)

#         # calculate stft (n_fft//2 + 1) - features
#         spec = np.abs(librosa.spectrum.stft(audio, win_length=frame_size, n_fft=n_fft, hop_length=hop_size, window="hann"))
#         power_spec = np.power(spec, 2)

#         # calculate mel spectrogram (n_mels) - features
#         mel_spec = librosa.feature.melspectrogram(S=power_spec, sr=sr, n_fft=frame_size, n_mels=n_mels)

#         # calculate mfccs (n_mfcc) - features
#         mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

#         # calculate chroma (n_chroma) - features
#         chroma = librosa.feature.chroma_stft(S=spec, sr=sr, n_chroma=n_chroma)

#         # calculate the tonnetz (perfect 5th, minor and major 3rd all in 2 d) - 6
#         tonnetz = librosa.feature.tonnetz(y=audio, sr=sr, chroma=chroma)

#         # calculate spectral contrast - (spectral_contrast_bands + 1)
#         spectral_contrast = librosa.feature.spectral_contrast(S=spec, sr=sr, n_bands=spectral_contrast_bands)

#         # initialize array
#         offset = 0
#         feature_count = 6
#         frame_count = spec.shape[1]
#         arr = np.zeros((feature_count, frame_count))

#         # calculate tempo - 1
#         onset_env = librosa.onset.onset_strength(audio, sr=sr)
#         tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
#         arr[offset:offset+1] = tempo
#         offset += 1

#         # calculate spectral centroid - 1
#         spectral_centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)
#         arr[offset:offset+1] = spectral_centroid
#         offset += 1

#         # calculate spectral bandwidth - 1
#         spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sr)
#         arr[offset:offset+1] = spectral_bandwidth
#         offset += 1

#         # calculate spectral flatness - 1
#         spectral_flatness = librosa.feature.spectral_flatness(S=spec)
#         arr[offset:offset+1] = spectral_flatness
#         offset += 1

#         # calculate spectral rolloff - 1
#         spectral_rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.85)
#         arr[offset:offset+1] = spectral_rolloff
#         offset += 1

#         # calculate zero crossing rate - 1
#         zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_size, hop_length=hop_size)
#         arr[offset:offset+1] = zcr

#         ret = {
#             'spec': torch.tensor(spec, dtype=torch.float32),
#             'mel_spec': torch.tensor(mel_spec, dtype=torch.float32),
#             'mfccs': torch.tensor(mfccs, dtype=torch.float32),
#             'chroma': torch.tensor(chroma, dtype=torch.float32),
#             'tonnetz': torch.tensor(tonnetz, dtype=torch.float32),
#             'spectral_contrast': torch.tensor(spectral_contrast, dtype=torch.float32)
#         }

#         return {
#             **ret,
#             'spectral_aggregate': torch.tensor(arr, dtype=torch.float32)
#         }

#     def get_features(self, info, args):

#         key = path.join(self.temp_folder, "{}-{}.pkl".format(info[self.audio_index], args))
#         if path.exists(key):
#             out = pickle.load(open(key, mode="rb"))
#             return out

#         x, sr = self.get_audio(info, args)
#         audio_out = self.preprocess_audio(x, sr)

#         features_out = self.__compute_features(
#             np.array(torch.squeeze(audio_out)),
#             duration=self.chunk_duration,
#             sample_rate=self.sample_rate,
#             **self.extractor_config
#         )


#         out = {
#             'raw': audio_out,
#             **features_out
#         }

#         pickle.dump(out, open(key, mode="wb"))

#         return out

#     def get_features(self, info, args):

#         x, sr = self.get_audio(info, args)
#         audio_out = self.preprocess_audio(x, sr)

#         features_out = self.__compute_features(
#             np.array(torch.squeeze(audio_out)), frame_size=self.frame_size, hop_size=self.hop_size, sample_rate=self.sample_rate)


#         out = (
#             audio_out,
#             *features_out
#         )

#         return out

# class GenericPickleDataset(Dataset):

#     def __calculate_count(self, meta, chunk_duration, overlap):
#         self.frames = []
#         row_i = 0
#         for (i, row) in meta.iterrows():
#             duration = row['duration']
#             n_frames = int(np.floor((duration - 1) / (chunk_duration - overlap)))
#             if n_frames == 0 and chunk_duration <= duration:
#                 self.frames.append((row_i, 0))
#                 continue
#             for j in range(n_frames):
#                 self.frames.append((row_i, j))
#             row_i += 1
#         return len(self.frames)

#     def __init__(self, meta_file, audio_index, label_index, chunk_duration, overlap, pickle_folder):

#         self.audio_index = audio_index
#         self.label_index = label_index
#         self.meta = pd.read_json(meta_file)
#         self.chunk_duration = chunk_duration
#         self.overlap = overlap
#         self.count = self.__calculate_count(self.meta, self.chunk_duration, self.overlap)
#         self.labels = np.array(
#             list(map(lambda x: self.meta.iloc[x[0]][self.label_index], self.frames))
#         )

#         self.pickle_folder = pickle_folder

#     def get_labels(self):
#         return self.labels

#     def get_label(self, info, args):
#         if type(self.label_index) is list:
#             return info[self.label_index].to_numpy()
#         return info[self.label_index]

#     def __len__(self):
#         return self.count

#     def __getitem__(self, index):
#         (info, args) = self.get_info(index)
#         X = self.get_features(info, args)
#         y = self.get_label(info, args)
#         return (X, y)

#     def get_info(self, index):
#         (meta_index, frame) = self.frames[index]
#         info = self.meta.iloc[meta_index]
#         return (info, frame)

#     def get_features(self, info, args):

#         key = path.join(self.pickle_folder, "{}-{}.pkl".format(info[self.audio_index], args))
#         if path.exists(key):
#             out = pickle.load(open(key, mode="rb"))
#             return out

#         raise FileNotFoundError("pickle file not found!")
