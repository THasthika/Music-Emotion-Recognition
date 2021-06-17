from torch.utils.data.dataloader import DataLoader
from models.revamped import Audio1DConvCat

from data import BaseAudioOnlyChunkedDataset
from torch.utils.data import Subset

import pytorch_lightning as pl


model = Audio1DConvCat(
    raw_audio_extractor_units=[1, 64, 128],
    classifier_units=[
        512,
        256,
        128
    ],
    n_classes=4)

data = BaseAudioOnlyChunkedDataset(
    "/storage/s3/splits/mer-taffc-kfold/train.json",
    "/storage/s3/raw/mer-taffc/audio"
)

train_indices = [x for x in range(0, int(len(data) * 0.7))]
val_indices = [x for x in range(int(len(data) * 0.7), len(data))]

train_ds = Subset(data, train_indices)
val_ds = Subset(data, val_indices)

test_data = BaseAudioOnlyChunkedDataset(
    "/storage/s3/splits/mer-taffc-kfold/train.json",
    "/storage/s3/raw/mer-taffc/audio"
)

train_dl = DataLoader(train_ds, batch_size=32, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=32, num_workers=4)

trainer = pl.Trainer(gpus=-1)

trainer.fit(model, train_dl, val_dataloaders=val_dl)