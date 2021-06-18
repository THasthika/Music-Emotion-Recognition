import yaml
import sys
from os import path

import kfold

def run(config_file):
    config = yaml.load(open(config_file, mode="r"), Loader=yaml.FullLoader)

    modelMod = __import__('models.revamped', fromlist=[config['model']['class']])
    ModelClass = getattr(modelMod, config['model']['class'])
    modelConfig = config['model']['config']

    dataMod = __import__('data', fromlist=[config['dataset']['class']])
    DatasetClass = getattr(dataMod, config['dataset']['class'])
    datasetConfig = config['dataset']['config']

    dataDir = config['dataset']['data_dir']
    splitDir = config['dataset']['split_dir']

    kfoldConfig = {}
    try:
        kfoldConfig = config['kfold']
        if not type(kfoldConfig) is dict:
            print("not a dictionary... ignoring")
            kfoldConfig = {}
    except:
        pass

    default_kfold_args = {
        'n_splits': 5,
        'stratify': False,
        'batch_size': 16,
        'num_workers': 4,
        'max_runs': None,
        'wandb_group': None,
        'wandb_project_name': None,
        'wandb_tags': None,
        'model_monitor': 'val/loss',
        'early_stop_monitor': 'val/acc',
        'early_stop_mode': 'max'
    }

    default_kfold_args.update(kfoldConfig)

    if default_kfold_args['wandb_project_name'] is None:
        default_kfold_args['wandb_project_name'] = 'mer'

    model = ModelClass(**modelConfig)

    train_ds = DatasetClass(
        path.join(splitDir, "train.json"),
        dataDir,
        **datasetConfig
    )

    test_ds = DatasetClass(
        path.join(splitDir, "test.json"),
        dataDir,
        **datasetConfig
    )

    crossValidator = kfold.CrossValidator(**default_kfold_args)

    crossValidator.fit(model, train_ds, test_ds)

if __name__ == "__main__":
    config_file = "config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    run(config_file)