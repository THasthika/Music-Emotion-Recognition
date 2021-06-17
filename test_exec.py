from models.revamped.audio_1dconv_stat import Audio1DConvStat
from models.revamped import Audio1DConvCat
from data import BaseAudioOnlyChunkedDataset, AudioOnlyStaticAVValues

from torch.utils.data import DataLoader

import torchinfo

model = Audio1DConvStat()

ds = AudioOnlyStaticAVValues(
    "/storage/s3/splits/mer-taffc-kfold/train.json",
    "/storage/s3/raw/mer-taffc/audio/",
    temp_folder="/tmp/xxx"
)

dl = DataLoader(ds, batch_size=2, num_workers=4)

torchinfo.summary(model, input_size=(1, 1, 22050*10))

for (X, y) in dl:
    print(X.shape)
    print(y.shape)
    # (X, y) = (X.to("cuda"), y.to("cuda"))
    y_pred = model(X)
    print(y_pred.shape)
    # print("OK!")
    break