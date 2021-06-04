from os import path

import torch
from torch.utils.data import Dataset

import torchaudio

import pandas as pd
import numpy as np

class BaseDataset(Dataset):

    def __init__(self, meta_file):

        self.meta = self.__get_meta(meta_file)
    
    def __get_meta(self, meta_file):
        meta_ext = path.splitext(meta_file)[1]
        if meta_ext == ".json":
            return pd.read_json(meta_file)
        elif meta_ext == ".csv":
            return pd.read_csv(meta_file)
        else:
            raise Exception("Unknown File Extension {}".format(meta_ext))

    def get_features(self, info, args):
        raise NotImplementedError()

    def get_label(self, info, args):
        raise NotImplementedError()

    def get_info(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        (info, args) = self.get_info(index)
        X = self.get_features(info, args)
        y = self.get_label(info, args)
        return (X, y)

class BaseAudioDataset(BaseDataset):

    def __init__(self, meta_file, sample_rate=22050, max_duration=5):
        super().__init__(meta_file)

        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.frame_count = int(self.sample_rate * max_duration)

        self.count = len(self.meta)

    def __len__(self):
        return self.count

    def get_audio(self, info, args):
        raise NotImplementedError()

    def preprocess_audio(self, audio, sr):
        x = torch.mean(audio, 0, True)
        out = torch.zeros(1, self.frame_count)
        
        effects = [
          ["rate", f"{self.sample_rate}"]
        ]
        
        x, sr2 = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
        
        if self.frame_count >= x.shape[1]:
            out[:, :x.shape[1]] = x
        else:
            out[:, :] = x[:, :self.frame_count]

        # out = torch.squeeze(out)
        # out = torch.unsqueeze(out, dim=1)
        
        return out

    def get_features(self, info, args):
        x, sr = self.get_audio(info, args)
        out = self.preprocess_audio(x, sr)
        return out

    def get_info(self, index):
        return (self.meta.iloc[index], None)

class BaseChunkedAudioOnlyDataset(BaseAudioDataset):
    
    def __init__(self, meta_file, sample_rate=22050, chunk_duration=5, overlap=2.5):
        super().__init__(meta_file, sample_rate, 1)

        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.frame_count = int(self.sample_rate * self.chunk_duration)

        self.count = self.__calculate_count(self.meta, self.chunk_duration, self.overlap)

    def __calculate_count(self, meta, chunk_duration, overlap):
        self.frames = []
        row_i = 0
        for (i, row) in meta.iterrows():
            duration = row['duration']
            n_frames = int(np.floor((duration - 1) / (chunk_duration - overlap)))
            if n_frames == 0 and chunk_duration <= duration:
                self.frames.append((row_i, 0))
                continue
            for j in range(n_frames):
                self.frames.append((row_i, j))
            row_i += 1
        return len(self.frames)

    def __len__(self):
        return self.count

    def get_info(self, index):
        (meta_index, frame) = self.frames[index]
        info = self.meta.iloc[meta_index]
        return (info, frame)

class GenericStaticAudioOnlyDataset(BaseAudioDataset):

    def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, max_duration=30):
        super().__init__(meta_file, sample_rate, max_duration)

        self.data_dir = data_dir
        self.audio_index = audio_index
        self.audio_extension = audio_extension
        self.label_index = label_index

    def get_label(self, info, args):
        if type(self.label_index) is list:
            return info[self.label_index].to_numpy()
        return info[self.label_index]

    def get_audio(self, info, args):
        audio_file = path.join(self.data_dir, "{}.{}".format(info[self.audio_index], self.audio_extension))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate
        frames = int(sr * self.max_duration)
        x, sr = torchaudio.load(audio_file, num_frames=frames)
        return (x, sr)

class GenericStaticChunkedAudioOnlyDataset(BaseChunkedAudioOnlyDataset):

    def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, chunk_duration=5, overlap=2.5):
        super().__init__(meta_file, sample_rate, chunk_duration, overlap)

        self.data_dir = data_dir
        self.audio_index = audio_index
        self.audio_extension = audio_extension
        self.label_index = label_index

    def get_label(self, info, args):
        if type(self.label_index) is list:
            return info[self.label_index].to_numpy()
        return info[self.label_index]

    def get_audio(self, info, args):
        audio_file = path.join(self.data_dir, "{}.mp3".format(info['song_id']))
        meta_data = torchaudio.info(audio_file)
        sr = meta_data.sample_rate

        frame = args
        offset = int(sr * self.overlap * frame)
        frames = int(sr * self.chunk_duration)

        x, sr = torchaudio.load(audio_file, frame_offset=offset, num_frames=frames)
        return (x, sr)

class GenericStaticAudioFeatureOnlyDataset(BaseAudioDataset):
    
    def __init__(self, meta_file, data_dir, audio_index='song_id', label_index='quadrant'):
        super().__init__(meta_file)
        self.data_dir = data_dir
        self.audio_index = audio_index
        self.label_index = label_index
        self.count = len(self.meta)

    def __len__(self):
        return self.count

    def get_info(self, index):
        info = self.meta.iloc[index]
        return (info, None)

    def __stack_matching_keys(self, kset, d):
        kset_out = np.array(d[kset[0]], dtype='float32')
        for k in kset:
            if k == kset[0]:
                continue
            kset_out = np.hstack((kset_out, d[k]))
        return kset_out

    def __pad_array(self, arr, size):
        out = np.zeros((size, arr.shape[1]), dtype='float32')
        if size >= arr.shape[0]:
            out[:arr.shape[0], :] = arr
        else:
            out[:, :] = arr[:size, :]
        return out
    
    def get_features(self, info, args):
        name = info[self.audio_index]
        fname = "{}.npy".format(name)
        fpath = path.join(self.data_dir, fname)
        d = np.load(fpath, allow_pickle=True)
        d = d[()]

        kset_1 = [
            'mfcc',
            'mfcc_bands',
            'mfcc_bands_log',
            'zero_crossing_rate',
            'spectral_centroid',
        ]

        kset_2 = [
            'chromagram'
        ]

        kset_3 = [
            'dancability'
        ]

        kset1_out = self.__stack_matching_keys(kset_1, d)
        kset2_out = self.__stack_matching_keys(kset_2, d)
        kset3_out = self.__stack_matching_keys(kset_3, d)

        kset1_out = self.__pad_array(kset1_out, 1294)
        kset2_out = self.__pad_array(kset2_out, 40)
        kset3_out = np.reshape(kset3_out, (1, 1))

        return (kset1_out, kset2_out, kset3_out)
    
    def get_label(self, info, args):
        if type(self.label_index) is list:
            return info[self.label_index].to_numpy()
        return info[self.label_index]

class GenericStaticHybridAudioOnlyDataset(GenericStaticAudioOnlyDataset):

    def __init__(self, meta_file, data_dir, audio_index='song_id', audio_extension='mp3', label_index='quadrant', sample_rate=22050, max_duration=30):
        super().__init__(meta_file, path.join(data_dir, "audio"), audio_index, audio_extension, label_index, sample_rate, max_duration)

        self.feature_dir = path.join(data_dir, "features")

    def __stack_matching_keys(self, kset, d):
        kset_out = np.array(d[kset[0]], dtype='float32')
        for k in kset:
            if k == kset[0]:
                continue
            kset_out = np.hstack((kset_out, d[k]))
        return kset_out

    def __pad_array(self, arr, size):
        out = np.zeros((size, arr.shape[1]), dtype='float32')
        if size >= arr.shape[0]:
            out[:arr.shape[0], :] = arr
        else:
            out[:, :] = arr[:size, :]
        return out

    def get_extracted_features(self, info, args):
        name = info[self.audio_index]
        fname = "{}.npy".format(name)
        fpath = path.join(self.feature_dir, fname)
        d = np.load(fpath, allow_pickle=True)
        d = d[()]

        kset_1 = [
            'mfcc',
            'mfcc_bands',
            'mfcc_bands_log',
            'zero_crossing_rate',
            'spectral_centroid',
        ]

        kset_2 = [
            'chromagram'
        ]

        kset_3 = [
            'dancability'
        ]

        kset1_out = self.__stack_matching_keys(kset_1, d)
        kset2_out = self.__stack_matching_keys(kset_2, d)
        kset3_out = self.__stack_matching_keys(kset_3, d)

        kset1_out = self.__pad_array(kset1_out, 1294)
        kset2_out = self.__pad_array(kset2_out, 40)
        kset3_out = np.reshape(kset3_out, (1, 1))

        return (kset1_out, kset2_out, kset3_out)

    def get_features(self, info, args):

        x, sr = self.get_audio(info, args)
        audio_out = self.preprocess_audio(x, sr)

        features_out = self.get_extracted_features(info, args)

        out = (
            audio_out,
            *features_out
        )

        return out