from sweeps import makeSweepTrainer
from models import ModelCatA as Model

CMDS = Model.CMDS

DEFAULT_DATA_ARTIFACT = "mer-taffc:latest"
DEFAULT_SPLIT_ARTIFACT = "mer-taffc-train-70-val-20-test-10-seed-42:latest"
DEFAULT_CONFIG = dict(map(lambda x: (x[0], x[2]), CMDS))

makeTrainer = makeSweepTrainer(Model, default_config=DEFAULT_CONFIG, monitor='val/loss')

def get_parse_args():
    return (DEFAULT_DATA_ARTIFACT, DEFAULT_SPLIT_ARTIFACT, CMDS)
