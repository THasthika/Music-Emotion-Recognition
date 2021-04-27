import os
import sys
import argparse

from dotenv import load_dotenv

from sweeps.mer_taffc import catA_mer_taffc
from sweeps.mer_taffc import catB_mer_taffc

from sweeps.deam import catA_deam
from sweeps.deam import catB_deam

from sweeps.deam import statA_deam
from sweeps.deam import statB_deam
from sweeps.deam import statC_deam

from sweeps.pmemo import catA_pmemo

from sweeps.pmemo import statA_pmemo

load_dotenv()

PROJECT_NAME = os.environ['PROJECT_NAME'] if 'PROJECT_NAME' in os.environ else "mer"
DATA_ARTIFACT = os.environ['DATA_ARTIFACT'] if 'DATA_ARTIFACT' in os.environ else None
SPLIT_ARTIFACT = os.environ['SPLIT_ARTIFACT'] if 'SPLIT_ARTIFACT' in os.environ else None
BATCH_SIZE = int(os.environ['BATCH_SIZE']) if 'BATCH_SIZE' in os.environ else 64
FIND_BATCH_SIZE = bool(os.environ['FIND_BATCH_SIZE']) if 'FIND_BATCH_SIZE' in os.environ else False
FIND_LR = bool(os.environ['FIND_LR']) if 'FIND_LR' in os.environ else False

_COMMON_ARGS = [
    ("project_name", str, PROJECT_NAME),
    ("data_artifact", str, DATA_ARTIFACT),
    ("split_artifact", str, SPLIT_ARTIFACT),
    ("batch_size", int, BATCH_SIZE),
    ("find_batch_size", bool, FIND_BATCH_SIZE),
    ("find_lr", bool, FIND_LR)
]

_MAPPING = {

    'catA_mer_taffc': catA_mer_taffc,
    'catB_mer_taffc': catB_mer_taffc,

    'catA_deam': catA_deam,
    'catB_deam': catB_deam,

    'statA_deam': statA_deam,
    'statB_deam': statB_deam,
    'statC_deam': statC_deam,

    'catA_pmemo': catA_pmemo,

    'statA_pmemo': statA_pmemo
}

def run_sweep(name):
    sweep = _MAPPING[name]

    ## init parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    sweep_parser = subparsers.add_parser(name)
    (data_artifact, split_artifact, cmds) = sweep.get_parse_args()

    model_config_keys = list(map(lambda x: x[0], cmds))

    cmds.extend(_COMMON_ARGS)

    for cmd in cmds:
        sweep_parser.add_argument("--{}".format(cmd[0]), type=cmd[1], default=cmd[2])

    args = parser.parse_args()
    options = vars(args)
    if options['data_artifact'] is None:
        options['data_artifact'] = data_artifact
    if options['split_artifact'] is None:
        options['split_artifact'] = split_artifact

    general_config = {}
    model_config = {}

    for x in options:
        if x in model_config_keys:
            model_config[x] = options[x]
        else:
            general_config[x] = options[x]

    train = sweep.makeTrainer(**general_config)

    train(config=model_config)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Should pass a sweep name")
    sweep_name = sys.argv[1]
    run_sweep(sweep_name)
