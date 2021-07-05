import pickle
import random
from os import path
import os
import string

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from data import *

class BaseDataset(Dataset):

    def __get_temp_folder(self, length=8):
        fname = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=length))
        return path.join("/tmp/", fname)

    def __check_cache_and_get_features(self, info, args):
        key = self.get_key(info, args)
        fkey = path.join(self.temp_folder, "{}.pkl".format(key))
        if (not self.force_compute) and path.exists(fkey):
            try:
                X = pickle.load(open(fkey, mode="rb"))
                return X
            except:
                print("Warning: failed to load pickle file. getting features... {}".format(fkey))
        X = self.get_features(info, args)
        pickle.dump(X, open(fkey, mode="wb"))
        return X

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

    def get_labels(self):
        return self.meta[QUADRANT]

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

    def get_labels(self):
        return self.labels

    def __init__(self, meta_file, chunk_duration=5, overlap=2.5, temp_folder=None, force_compute=False):
        super().__init__(meta_file, temp_folder=temp_folder, force_compute=force_compute)

        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.frames = self.__calculate_frames(
            self.meta, self.chunk_duration, self.overlap)
        self.count = len(self.frames)


        self.labels = np.array(
            list(map(lambda x: self.meta.iloc[x[0]][QUADRANT], self.frames))
        )

    def get_info(self, index):
        (meta_index, frame) = self.frames[index]
        info = self.meta.iloc[meta_index]
        return (info, frame)