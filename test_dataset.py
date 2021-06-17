from data import AudioOnlyStaticQuadrantAndAVValues
from torch.utils.data import DataLoader

META_FILE = "/storage/s3/splits/mer-taffc-kfold/train.json"
DATA_DIR = "/storage/s3/raw/mer-taffc/audio/"
TEMP_FOLDER = "/storage/s3/precomputed/mer-taffc/22050-10-5"

ds = AudioOnlyStaticQuadrantAndAVValues(META_FILE, DATA_DIR, temp_folder=TEMP_FOLDER)
dl = DataLoader(ds, batch_size=8, num_workers=4)


for (X, y) in dl:
    print(X)
    print(y)
    break