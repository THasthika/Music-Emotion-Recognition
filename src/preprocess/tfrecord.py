import pandas as pd
import numpy as np
import os

_SEED = 2020

class TFRecordsConverter:
    """Convert Audio files into TFRecords with corresponding labels"""

    # When compression is used, resulting TFRecord files are four to five times
    # smaller. So, we can reduce the number of shards by this factor
    _COMPRESSION_SCALING_FACTOR = 4

    def __init__(self, meta_path, output_dir, n_shards_train=None, n_shards_test=None, n_shards_val=None, duration=10, sample_rate=22050, test_size=0.1, val_size=0.1):
        self.output_dir = output_dir
        self.duration = duration
        self.sample_rate = sample_rate

        df = pd.read_csv(meta_path, index_col=0)
        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=_SEED)

        n_samples = len(df)
        self.n_test = np.ceil(n_samples * test_size)
        self.n_val = np.ceil(n_samples * val_size)
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