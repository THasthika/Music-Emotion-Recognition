from os import path
from posixpath import split


import pytorch_lightning as pl

from torch.utils.data import DataLoader

class BaseModel(pl.LightningModule):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                dataset_class=None,
                dataset_class_args={
                    'sample_rate': 22050,
                    'max_duration': 30,
                    'label_type': 'categorical',
                    'audio_dir': ''
                },
                split_dir=None):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_class = dataset_class
        self.dataset_class_args = dataset_class_args
        self.split_dir = split_dir

    def set_model_parameter(self, config, config_keys, default):
        if not (type(config_keys) is list or type(config_keys) is tuple):
            config_keys = [config_keys]
        exists = True
        temp = []
        for k in config_keys:
            if not k in config:
                exists = False
                break
            temp.append(config[k])
        if not exists:
            return default
        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def prepare_data(self):

        if self.dataset_class is None or self.split_dir is None:
            return

        split_dir = self.split_dir
        DSClass = self.dataset_class
        additional_args = self.dataset_class_args

        train_meta_file = path.join(split_dir, "train.json")
        val_meta_file = path.join(split_dir, "val.json")
        test_meta_file = path.join(split_dir, "test.json")

        has_val = False
        has_test = False
        if path.exists(val_meta_file):
            has_val = True
        if path.exists(test_meta_file):
            has_test = True

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_ds = DSClass(
            meta_file=train_meta_file,
            **additional_args)

        if has_val:
            self.val_ds = DSClass(
                meta_file=val_meta_file,
                **additional_args)

        if has_test:
            self.test_ds = DSClass(
                meta_file=test_meta_file,
                **additional_args)

    def train_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)