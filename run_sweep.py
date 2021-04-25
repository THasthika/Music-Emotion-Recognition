import sys
import argparse

from sweeps import catA_deam
from sweeps import catB_deam

from sweeps import catA_mer_taffc
from sweeps import catB_mer_taffc

from sweeps import statA_deam

_COMMON_ARGS = [
    ("project_name", str, "mer"),
    ("data_artifact", str, None),
    ("split_artifact", str, None),
    ("batch_size", int, 64),
    ("find_batch_size", bool, False),
    ("find_lr", bool, False)
]

_MAPPING = {
    'catA_deam': catA_deam,
    'catB_deam': catB_deam,
    
    'catA_mer_taffc': catA_mer_taffc,
    'catB_mer_taffc': catB_mer_taffc,

    'statA_deam': statA_deam
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
