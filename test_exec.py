from models.revamped import Audio1DConvCat
from data import BaseAudioOnlyChunkedDataset

from torch.utils.data import DataLoader

model = Audio1DConvCat().cuda()

ds = BaseAudioOnlyChunkedDataset(
    "/storage/s3/splits/mer-taffc-kfold/train.json",
    "/storage/s3/raw/mer-taffc/audio/",
    temp_folder="/tmp/xxx"
)

dl = DataLoader(ds, batch_size=32, num_workers=4)

for (X, y) in dl:
    (X, y) = (X.to("cuda"), y.to("cuda"))
    y_pred = model(X)
    print("OK!")