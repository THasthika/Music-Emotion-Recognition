# Preprocessing File for PMEmo2019 dataset
# creates static and dynamic datasets for the PMEmo
##

from os import path
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

import librosa

PER_FILE_ENTRIES = 100

BASE_DIR = "datasets/PMEmo2019"

AUDIO_DIR = path.join(BASE_DIR, "chorus")
ANNOTATION_DIR = path.join(BASE_DIR, "annotations")

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def readAudio(musicId):
    apath = path.join(AUDIO_DIR, "{}.mp3".format(musicId))

    audio, sr = librosa.core.load(apath, mono=True, sr=44100)

    return audio, sr


def split_dataset(data: pd.DataFrame, test_percentage=0.2):

    if test_percentage > 1.0:
        raise "Invalid percentages"

    train_percentage = 1.0 - test_percentage

    count = len(data)

    train_si = 0
    train_ei = (train_si + math.ceil(count * train_percentage))

    test_si = train_ei
    test_ei = count

    train_data = data.iloc[train_si:train_ei]
    test_data = data.iloc[test_si:test_ei]

    return (train_data, test_data)


def MER_static_example(arousal_mean: float, arousal_std: float, valence_mean: float, valence_std: float, song_id: str):
    feature = {
        'arousal_mean': _float_feature([arousal_mean]),
        'arousal_std': _float_feature([arousal_std]),
        'valence_mean': _float_feature([valence_mean]),
        'valence_std': _float_feature([valence_std]),
        'song_id': _bytes_feature(song_id)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def MER_dynamic_example(arousal_mean: list, arousal_std: list, valence_mean: list, valence_std: list, timestamps: list, song_id: str):
    feature = {
        'arousal_mean': _float_feature(arousal_mean),
        'arousal_std': _float_feature(arousal_std),
        'valence_mean': _float_feature(valence_mean),
        'valence_std': _float_feature(valence_std),
        'timestamps': _float_feature(timestamps),
        'song_id': _bytes_feature(song_id)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

### working on the static annotations

def prepare_static_dataset():

    output_file="pmemo_static.tfrecord"

    csv_path = path.join(ANNOTATION_DIR, "static_annotations_all.csv")

    all_df = pd.read_csv(csv_path)

    all_df[["musicId"]] = all_df[["musicId"]].astype(str)

    print(all_df)

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(output_file, options=options) as writer:
        for index, row in all_df.iterrows():
            s_id = row.iloc[0]
            ar_m = row.iloc[1]
            ar_s = row.iloc[2]
            va_m = row.iloc[3]
            va_s = row.iloc[4]

            x = MER_static_example(ar_m, ar_s, va_m, va_s, s_id)
            writer.write(x.SerializeToString())

            print("{} done...".format(index))

def change_range_column(df: pd.DataFrame, cols, curr_min: float, curr_max: float, n_min: float, n_max: float):
    col = df.iloc[:, cols]
    n_col = (((col - curr_min) * (n_max - n_min)) / (curr_max - curr_min)) + n_min
    return n_col.round(5)

def merge_static_dataframes():
    std_csv_path = path.join(ANNOTATION_DIR, "static_annotations_std.csv")
    mean_csv_path = path.join(ANNOTATION_DIR, "static_annotations.csv")
    
    std_df = pd.read_csv(std_csv_path)
    mean_df = pd.read_csv(mean_csv_path)

    all_df = mean_df.merge(std_df, how="inner", on=["musicId"])

    cols = all_df.columns.tolist()
    cols = cols[0:2] + cols[3:4] + cols[2:3] + cols[4:5]
    all_df = all_df[cols]

    all_df[["musicId"]] = all_df[["musicId"]].astype(str)

    n_col = change_range_column(all_df, [1, 3], 0, 1, -1, 1)
    all_df.iloc[:, 1] = n_col.iloc[:, 0]
    all_df.iloc[:, 3] = n_col.iloc[:, 1]

    all_df.to_csv(path.join(ANNOTATION_DIR, "static_annotations_all.csv"), index=False)

# def merge_dynamic_dataframes9):
#     std_csv_path = path.join(ANNOTATION_DIR, "static_annotations_std.csv")
#     mean_csv_path = path.join(ANNOTATION_DIR, "static_annotations.csv")

# merge_static_dataframes()

# prepare_static_dataset()

# # tf.io.read_file()
# # tfio.audio.decode_mp3()

audio_binary = tf.io.read_file(path.join(AUDIO_DIR, "1.mp3"))
x = tfio.audio.decode_mp3(audio_binary)
print(x)
# print(tfio.audio.decode_mp3)

# (audio, sr) = tfio.audio.decode_mp3()
# print(sr)
# # y = tf.math.reduce_mean(x, axis=0)
# print(y)

# (train_df, valid_df, test_df) = split_dataset(all_df)

# all_df[[0]]



# record_file = "pmemo_static.tfrecords"
# n_samples = len(all_df)

# n_files = math.ceil(n_samples / (1.0 * PER_FILE_ENTRIES))



# options = tf.io.TFRecordOptions(compression_type="GZIP")
# with tf.io.TFRecordWriter(record_file, options=options) as writer:

#     for index, row in all_df.iterrows():
#         audio, sr = readAudio(row["musicId"])

#         ar_m = row.iloc[1]
#         ar_s = row.iloc[2]
#         va_m = row.iloc[3]
#         va_s = row.iloc[4]

#         x = MER_static_example(ar_m, ar_s, va_m, va_s, audio)
#         writer.write(x.SerializeToString())

#         print("{} done...".format(index))
#         break

# audio_data = np.array(audio_data)

# all_df[["audio"]] = audio_data

# print(all_df)

# all_df.to_csv("./test.csv")



## For the dynamic situation take input of varying clips