from os import path

import wandb

import pytorch_lightning as pl

from data import DeamDataset, MERTaffcDataset, PMEmoDataset, FMADataset, FSD50KDataset

from torch.utils.data import DataLoader

class BaseModel(pl.LightningModule):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                sample_rate=22050,
                chunk_duration=5,
                overlap=2.5,
                label_type="categorical"):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            num_workers (int, optional): [description]. Defaults to 4.
            sample_rate (int, optional): The sample rate of audio.
            chunk_duration (float, optional): The duration to take from the audio.
            overlap (float, optiona): The overlap on frames.
            label_type (str, optional): "categorical", "static", "dynamic". Defauts to "categorical"
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.label_type = label_type

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap

    def _get_data_dir(self):
        raise NotImplementedError()

    def _get_split_dir(self):
        raise NotImplementedError()

    def _get_ds_class(self):
        raise NotImplementedError()

    def prepare_data(self):

        split_dir = self._get_split_dir()
        data_dir = self._get_data_dir()
        DSClass = self._get_ds_class()

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

        self.train_ds = DSClass(
            audio_dir=audio_dir,
            meta_file=train_meta_file,
            label_type=self.label_type,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            overlap=self.overlap)

        if has_val:
            self.val_ds = DSClass(
                audio_dir=audio_dir,
                meta_file=val_meta_file,
                label_type=self.label_type,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                overlap=self.overlap)

        if has_test:
            self.test_ds = DSClass(
                audio_dir=audio_dir,
                meta_file=test_meta_file,
                label_type=self.label_type,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                overlap=self.overlap)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds is None: return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_ds is None: return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

class WandbBaseModel(BaseModel):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                sample_rate=22050,
                chunk_duration=5,
                overlap=2.5,
                data_artifact=None,
                split_artifact=None,
                label_type="categorical"):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            num_workers (int, optional): [description]. Defaults to 4.
            sample_rate (int, optional): The sample rate of audio.
            chunk_duration (float, optional): The duration to take from the audio.
            overlap (float, optiona): The overlap on frames.
            data_artifact (str, optional): [description]. Defaults to None.
            split_artifact (str, optional): [description]. Defaults to None.
            label_type (str, optional): "categorical", "static", "dynamic". Defauts to "categorical"
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap,
            label_type=label_type)

        self.data_artifact = data_artifact
        self.split_artifact = split_artifact

    def _get_data_dir(self):
        data_at = wandb.use_artifact(self.data_artifact, type="data-raw")
        data_dir = data_at.download()
        return data_dir

    def _get_split_dir(self):
        split_at = wandb.use_artifact(self.split_artifact, type="data-split")
        split_dir = split_at.download()
        return split_dir

    def _get_ds_class(self):
        DSClass = None
        if self.data_artifact.startswith("mer-taffc"):
            DSClass = MERTaffcDataset
        elif self.data_artifact.startswith("deam"):
            DSClass = DeamDataset
        elif self.data_artifact.startswith("pmemo"):
            DSClass = PMEmoDataset
        return DSClass

class FolderBaseModel(BaseModel):
    def __init__(self,
                batch_size=32,
                num_workers=4,
                sample_rate=22050,
                chunk_duration=5,
                overlap=2.5,
                data_dir=None,
                split_dir=None,
                label_type="categorical"):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            num_workers (int, optional): [description]. Defaults to 4.
            sample_rate (int, optional): The sample rate of audio.
            chunk_duration (float, optional): The duration to take from the audio.
            overlap (float, optiona): The overlap on frames.
            data_dir (str, optional): [description]. Defaults to None.
            split_dir (str, optional): [description]. Defaults to None.
            label_type (str, optional): "categorical", "static", "dynamic". Defauts to "categorical"
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap,
            label_type=label_type)

        self.data_dir = data_dir
        self.split_dir = split_dir

    def _get_data_dir(self):
        return self.data_dir

    def _get_split_dir(self):
        return self.split_dir

    def _get_ds_class(self):
        DSClass = None
        if "mer-taffc" in self.data_dir:
            DSClass = MERTaffcDataset
        elif "deam" in self.data_dir:
            DSClass = DeamDataset
        elif "pmemo" in self.data_dir:
            DSClass = PMEmoDataset
        elif "fsd50k" in self.data_dir:
            DSClass = FSD50KDataset
        elif "fma" in self.data_dir:
            DSClass = FMADataset
        return DSClass