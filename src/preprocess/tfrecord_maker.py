import argparse
import math
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_META_CSV = os.path.join(_BASE_DIR, 'meta.csv')
_DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, 'tfrecords')

_DEFAULT_DURATION = 10  # seconds
_DEFAULT_SAMPLE_RATE = 22050

_DEFAULT_TEST_SIZE = 0.1
_DEFAULT_VAL_SIZE = 0.1

_SEED = 2020


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parallelize(func, data):
    processes = cpu_count() - 1
    with Pool(processes) as pool:
        # We need the enclosing list statement to wait for the iterator to end
        # https://stackoverflow.com/a/45276885/1663506
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))


class TFRecordsConverter:
    """Convert WAV files to TFRecords."""

    # When compression is used, resulting TFRecord files are four to five times
    # smaller. So, we can reduce the number of shards by this factor
    _COMPRESSION_SCALING_FACTOR = 4

    def __init__(self, meta, output_dir, n_shards_train, n_shards_test,
                 n_shards_val, duration, sample_rate, test_size, val_size):
        self.output_dir = output_dir
        self.duration = duration
        self.sample_rate = sample_rate

        df = pd.read_csv(meta, index_col=0)
        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=_SEED)

        n_samples = len(df)
        self.n_test = math.ceil(n_samples * test_size)
        self.n_val = math.ceil(n_samples * val_size)
        self.n_train = n_samples - self.n_test - self.n_val

        if n_shards_train is None or n_shards_test is None or n_shards_val is None:
            self.n_shards_train = self._n_shards(self.n_train)
            self.n_shards_test = self._n_shards(self.n_test)
            self.n_shards_val = self._n_shards((self.n_val))
        else:
            self.n_shards_train = n_shards_train
            self.n_shards_test = n_shards_test
            self.n_shards_val = n_shards_val

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __repr__(self):
        return ('{}.{}(output_dir={}, n_shards_train={}, n_shards_test={}, '
                'n_shards_val={}, duration={}, sample_rate={}, n_train={}, '
                'n_test={}, n_val={})').format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.output_dir,
            self.n_shards_train,
            self.n_shards_test,
            self.n_shards_val,
            self.duration,
            self.sample_rate,
            self.n_train,
            self.n_test,
            self.n_val,
        )

    def _n_shards(self, n_samples):
        """Compute number of shards for number of samples.
        TFRecords are split into multiple shards. Each shard's size should be
        between 100 MB and 200 MB according to the TensorFlow documentation.
        Parameters
        ----------
        n_samples : int
            The number of samples to split into TFRecord shards.
        Returns
        -------
        n_shards : int
            The number of shards needed to fit the provided number of samples.
        """
        return math.ceil(n_samples / self._shard_size())

    def _shard_size(self):
        """Compute the shard size.
        Computes how many WAV files with the given sample-rate and duration
        fit into one TFRecord shard to stay within the 100 MB - 200 MB limit.
        Returns
        -------
        shard_size : int
            The number samples one shard can contain.
        """
        shard_max_bytes = 200 * 1024**2  # 200 MB maximum
        audio_bytes_per_second = self.sample_rate * 2  # 16-bit audio
        audio_bytes_total = audio_bytes_per_second * self.duration
        shard_size = shard_max_bytes // audio_bytes_total
        return shard_size * self._COMPRESSION_SCALING_FACTOR

    def _write_tfrecord_file(self, shard_data):
        """Write TFRecord file.
        Parameters
        ----------
        shard_data : tuple (str, list)
            A tuple containing the shard path and the list of indices to write
            to it.
        """
        shard_path, indices = shard_data
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for index in indices:
                file_path = self.df.file_path.iloc[index]
                label = self.df.label.iloc[index]

                raw_audio = tf.io.read_file(file_path)
                audio, sample_rate = tf.audio.decode_wav(
                    raw_audio,
                    desired_channels=1,  # mono
                    desired_samples=self.sample_rate * self.duration)

                # Example is a flexible message type that contains key-value
                # pairs, where each key maps to a Feature message. Here, each
                # Example contains two features: A FloatList for the decoded
                # audio data and an Int64List containing the corresponding
                # label's index.
                example = tf.train.Example(features=tf.train.Features(feature={
                    'audio': _float_feature(audio.numpy().flatten().tolist()),
                    'label': _int64_feature(label)}))

                out.write(example.SerializeToString())

    def _get_shard_path(self, split, shard_id, shard_size):
        """Construct a shard file path.
        Parameters
        ----------
        split : str
            The data split. Typically 'train', 'test' or 'validate'.
        shard_id : int
            The shard ID.
        shard_size : int
            The number of samples this shard contains.
        Returns
        -------
        shard_path : str
            The constructed shard path.
        """
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}.tfrec'.format(split, shard_id,
                                                        shard_size))

    def _split_data_into_shards(self):
        """Split data into train/test/val sets.
        Split data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        Returns
        -------
        shards : list [tuple]
            The shards as a list of tuples. Each item in this list is a tuple
            which contains the shard path and a list of indices to write to it.
        """
        shards = []

        splits = ('train', 'test', 'validate')
        split_sizes = (self.n_train, self.n_test, self.n_val)
        split_n_shards = (self.n_shards_train, self.n_shards_test,
                          self.n_shards_val)

        offset = 0
        for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
            print('Splitting {} set into TFRecord shards...'.format(split))
            shard_size = math.ceil(size / n_shards)
            cumulative_size = offset + size
            for shard_id in range(1, n_shards + 1):
                step_size = min(shard_size, cumulative_size - offset)
                shard_path = self._get_shard_path(split, shard_id, step_size)
                # Select a subset of indices to get only a subset of
                # audio-files/labels for the current shard.
                file_indices = np.arange(offset, offset + step_size)
                shards.append((shard_path, file_indices))
                offset += step_size

        return shards

    def convert(self):
        """Convert to TFRecords."""
        shard_splits = self._split_data_into_shards()
        _parallelize(self._write_tfrecord_file, shard_splits)

        print('Number of training examples: {}'.format(self.n_train))
        print('Number of testing examples: {}'.format(self.n_test))
        print('Number of validation examples: {}'.format(self.n_val))
        print('TFRecord files saved to {}'.format(self.output_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--meta-data-csv', type=str, dest='meta_csv',
                        default=_DEFAULT_META_CSV,
                        help='File containing audio file-paths and '
                             'corresponding labels. (default: %(default)s)')
    parser.add_argument('-o', '--output-dir', type=str, dest='output_dir',
                        default=_DEFAULT_OUTPUT_DIR,
                        help='Output directory to store TFRecord files.'
                             '(default: %(default)s)')
    parser.add_argument('--num-shards-train', type=int,
                        dest='n_shards_train',
                        help='Number of shards to divide training set '
                             'TFRecords into. Will be estimated from other '
                             'parameters if not explicitly set.')
    parser.add_argument('--num-shards-test', type=int,
                        dest='n_shards_test',
                        help='Number of shards to divide testing set '
                             'TFRecords into. Will be estimated from other '
                             'parameters if not explicitly set.')
    parser.add_argument('--num-shards-val', type=int,
                        dest='n_shards_val',
                        help='Number of shards to divide validation set '
                             'TFRecords into. Will be estimated from other '
                             'parameters if not explicitly set.')
    parser.add_argument('--duration', type=int,
                        dest='duration',
                        default=_DEFAULT_DURATION,
                        help='The duration for the resulting fixed-length '
                             'audio-data in seconds. Longer files are '
                             'truncated. Shorter files are zero-padded. '
                             '(default: %(default)s)')
    parser.add_argument('--sample-rate', type=int,
                        dest='sample_rate',
                        default=_DEFAULT_SAMPLE_RATE,
                        help='The _actual_ sample-rate of wav-files to '
                             'convert. Re-sampling is not yet supported. '
                             '(default: %(default)s)')
    parser.add_argument('--test-size', type=float,
                        dest='test_size',
                        default=_DEFAULT_TEST_SIZE,
                        help='Fraction of examples in the testing set. '
                             '(default: %(default)s)')
    parser.add_argument('--val-size', type=float,
                        dest='val_size',
                        default=_DEFAULT_VAL_SIZE,
                        help='Fraction of examples in the validation set. '
                             '(default: %(default)s)')

    return parser.parse_args()


def main(args):
    converter = TFRecordsConverter(args.meta_csv,
                                   args.output_dir,
                                   args.n_shards_train,
                                   args.n_shards_test,
                                   args.n_shards_val,
                                   args.duration,
                                   args.sample_rate,
                                   args.test_size,
                                   args.val_size)
    converter.convert()


if __name__ == '__main__':
    main(parse_args())