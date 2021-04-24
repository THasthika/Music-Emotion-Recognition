from os import path

import wandb

import pytorch_lightning as pl

from data import DeamDataset, MERTaffcDataset

from torch.utils.data import DataLoader

class BaseModel(pl.LightningModule):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                sample_rate=22050,
                duration=30,
                data_artifact=None,
                split_artifact=None,
                label_type="categorical"):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            num_workers (int, optional): [description]. Defaults to 4.
            sample_rate (int, optional): The sample rate of audio.
            duration (float, optional): The duration to take from the audio.
            data_artifact (str, optional): [description]. Defaults to None.
            split_artifact (str, optional): [description]. Defaults to None.
            label_type (str, optional): "categorical", "static", "dynamic". Defauts to "categorical"
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_artifact = data_artifact
        self.split_artifact = split_artifact
        self.label_type = label_type

        self.sample_rate = sample_rate
        self.duration = duration

    def prepare_data(self):
        split_at = wandb.use_artifact(self.split_artifact, type="data-split")
        split_dir = split_at.download()

        data_at = wandb.use_artifact(self.data_artifact, type="data-raw")
        data_dir = data_at.download()

        audio_dir = path.join(data_dir, "audio")
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

        DSClass = None
        if self.data_artifact.startswith("mer-taffc"):
            DSClass = MERTaffcDataset
        elif self.data_artifact.startswith("deam"):
            DSClass = DeamDataset

        self.train_ds = DSClass(
            audio_dir=audio_dir,
            meta_file=train_meta_file,
            label_type=self.label_type,
            sample_rate=self.sample_rate,
            duration=self.duration)

        if has_val:
            self.val_ds = DSClass(
                audio_dir=audio_dir,
                meta_file=val_meta_file,
                label_type=self.label_type,
                sample_rate=self.sample_rate,
                duration=self.duration)

        if has_test:
            self.test_ds = DSClass(
                audio_dir=audio_dir,
                meta_file=test_meta_file,
                label_type=self.label_type,
                sample_rate=self.sample_rate,
                duration=self.duration)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)