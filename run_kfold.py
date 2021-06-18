import yaml
import json
import sys
from os import path
import kfold

def __check_exists(d, k):
    if not k in d:
        return False
    if d[k] is None:
        return False
    return True

def __get_value(d: dict, k, default=None, raiseError=False):
    if __check_exists(d, k):
        return d[k]
    if not raiseError:
        return default
    raise KeyError("{} not found in {}".format(k, d.keys()))

def __check_dirs(*dirs):
    not_found_dirs = []
    for dir in dirs:
        if not path.exists(dir):
            not_found_dirs.append(dir)
    if len(not_found_dirs) > 0:
        raise FileNotFoundError("Could not find these file: {}".format(not_found_dirs))

def __print_configs(modelConfig, datasetConfig, kfoldConfig, dataDir, splitDir):
    print("--- MODEL CONFIG ---")
    print(json.dumps(modelConfig, sort_keys=False, indent=4))

    print("--- DATASET CONFIG ---")
    print("DATA_DIR: {}".format(dataDir))
    print("SPLIT_DIR: {}".format(splitDir))
    print(json.dumps(datasetConfig, sort_keys=False, indent=4))

    print("--- KFOLD CONFIG ---")
    print(json.dumps(kfoldConfig, sort_keys=False, indent=4))

def run(config_file):
    config = yaml.load(open(config_file, mode="r"), Loader=yaml.FullLoader)

    defaultConfig = yaml.load(open("default.yaml", mode="r"), Loader=yaml.FullLoader)

    debugMode = (__check_exists(config, 'debug') and config['debug'] == True)

    modelMod = __import__('models.revamped', fromlist=[config['model']['class']])
    ModelClass = getattr(modelMod, config['model']['class'])
    modelConfig = config['model']['config']


    ## dataset config
    dataMod = __import__('data', fromlist=[config['dataset']['class']])
    DatasetClass = getattr(dataMod, config['dataset']['class'])
    datasetConfig = config['dataset']['config']

    datasetName = config['dataset']['name']
    basePath = defaultConfig['base_path']

    dataDirPostfix = __get_value(config['dataset'], 'data_dir_postfix', default=None)
    dataDir = path.join(basePath, 'raw', datasetName)
    if not dataDirPostfix is None:
        dataDir = path.join(dataDir, dataDirPostfix)

    splitDir = path.join(basePath, 'splits',  __get_value(config['dataset'], 'split_dir', default="{}-kfold".format(datasetName)))

    duration = __get_value(config['dataset']['config'], 'chunk_duration', default=None)
    if duration is None:
        duration = __get_value(config['dataset']['config'], 'duration', raiseError=True)
    tempFolderName = "{}-{}".format(
        config['dataset']['config']['sr'],
        duration
    )

    if __check_exists(config['dataset']['config'], 'chunk_duration'):
        tempFolderName = path.join("chunked", tempFolderName + "-{}".format(config['dataset']['config']['overlap']))

    tempFolder = config['dataset']['config']['temp_folder'] if __check_exists(config['dataset']['config'], 'temp_folder') else  path.join(basePath, 'precomputed', datasetName, tempFolderName)
    datasetConfig['temp_folder'] = tempFolder

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

    ## add tags
    default_kfold_args['wandb_tags'].append("model:{}".format(config['model']['class']))
    default_kfold_args['wandb_tags'].append("dataset:{}".format(config['dataset']['name']))
    default_kfold_args['wandb_tags'].append("dataset-class:{}".format(config['dataset']['class']))
    default_kfold_args['wandb_tags'].append("sample-rate:{}".format(config['dataset']['config']['sr']))
    default_kfold_args['wandb_tags'].append("duration:{}".format(duration))
    default_kfold_args['wandb_tags'].append("chunked:{}".format('true' if __check_exists(config['dataset']['config'], 'chunk_duration') else 'false'))

    if debugMode:
        __print_configs(modelConfig, datasetConfig, default_kfold_args, dataDir, splitDir)

    __check_dirs(dataDir, splitDir)

    model = ModelClass(**modelConfig)
    print("Model Created...")

    train_ds = DatasetClass(
        path.join(splitDir, "train.json"),
        dataDir,
        **datasetConfig
    )
    print("Train Dataset Created...")

    test_ds = DatasetClass(
        path.join(splitDir, "test.json"),
        dataDir,
        **datasetConfig
    )
    print("Test Dataset Created...")

    crossValidator = kfold.CrossValidator(**default_kfold_args)
    print("Cross Validator Created...")

    crossValidator.fit(model, train_ds, test_ds)

if __name__ == "__main__":
    config_file = sys.argv[1]
    run(config_file)