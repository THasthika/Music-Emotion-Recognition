import json

from execs.a_1dconv import kfold_run as a1krun

MAPPING_EXECS = {
    "a_1dconv": a1krun
}

config = json.loads(open("config.json", mode="r"))
execs_name = config['execs']
model_config = config['model']
dataset_config = config['dataset']

fn = MAPPING_EXECS[execs_name]
fn(up_model_config=model_config, up_dataset_args=dataset_config)