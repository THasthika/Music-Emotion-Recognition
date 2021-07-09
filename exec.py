"""

python exec.py list runs [subpath] -> list all the runs or runs in subpath
python exec.py list models [subpath] -> list all the models with (name and versions available) optional subpath

python exec.py check (run_location) [--shape] -> check data loader and forward pass
    example -> python exec.py check run 1dconv.cat.a_1dconv_cat.r-01

python exec.py train (run_location) -> train run


"""

import os
from pprint import pprint

from torch.autograd.grad_mode import F

import data
from os import path, walk
import re

import yaml
import click

from torch.utils.data import DataLoader
import torch
import multiprocessing

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torchinfo
import wandb
from utils import kfold


def __load_yaml_file(file_path):
    return yaml.load(open(file_path, mode="r"), Loader=yaml.FullLoader)


def __load_default_config():
    config = __load_yaml_file(path.join(WORKING_DIR, "config.yaml"))
    return config


def __get_model_info(fp):
    _cn = list(filter(lambda x: x.startswith("class"),
               open(fp, mode="r").readlines()))[0]
    x = re.search("class ([^(_V)]+)_V([\d+])", _cn)
    model_version = int(x.group(2))
    model_name = x.group(1)
    return {'name': model_name, 'version': model_version}


def __get_gpu_count():
    if torch.cuda.is_available():
        return -1
    return None


def __get_num_workers():
    return multiprocessing.cpu_count()


def __get_wandb_tags(model_name, version, dataset, additional_tags=[]):
    return [
        'model:{}'.format(model_name),
        'dataset:{}'.format(dataset),
        'version:{}'.format(version),
        *additional_tags
    ]


def __parse_run_location(run):
    rp = run.replace(".", "/")
    _t = rp.split("/")
    if len(_t) > 3:
        rp = _t[:-1]
        rf = "{}.yaml".format(_t[-1])
    elif len(_t) <= 3:
        rp = _t
        rf = "default.yaml"
    rd = path.join(*rp)
    return (rd, rf)


def __is_subset(p, q):
    """
    check if p is a subset of q
    """
    return set(p).issubset(q)


def __load_data_class(run):

    run_s = run.split(".")

    if __is_subset(["stat", "acl"], run_s):
        from data.stat_multi import StatAudioLyricDataset
        return StatAudioLyricDataset
    
    if __is_subset(["cat", "acl"], run_s):
        from data.cat_multi import CatAudioLyricDataset
        return CatAudioLyricDataset

    if __is_subset(["stat"], run_s):
        from data.stat import StatAudioDataset
        return StatAudioDataset
    
    if __is_subset(["cat"], run_s):
        from data.cat import CatAudioDataset
        return CatAudioDataset

    raise ModuleNotFoundError("Unknown DataClass")


def __load_model_class(run, model_version):

    runp = run.split(".")
    if len(runp) > 3:
        runp = runp[:-1]

    model_file = "model_v{}.py".format(model_version)
    run_path = path.join(WORKING_DIR, "models", path.join(*runp), model_file)
    model_info = __get_model_info(run_path)
    ModelClsName = "{}_V{}".format(model_info['name'], model_info['version'])

    runp = ".".join(runp)
    pkg_path = "models.{}.model_v{}".format(runp, model_version)

    print("Loading Model Class {} from {}".format(ModelClsName, pkg_path))

    modelMod = __import__(pkg_path, fromlist=[ModelClsName])
    ModelClass = getattr(modelMod, ModelClsName)

    return (ModelClass, model_info)


def __is_kfold(data_config):
    return "kfold" in data_config['split']


def __parse_data_args(data_config):

    dataset_name = data_config['dataset']
    split_name = data_config['split']
    sub_folder = data_config['sub_folder'] if 'sub_folder' in data_config else None
    data_params = data_config['params'] if 'params' in data_config else {}

    temp_folder = path.join(TEMP_DIR, data_config['temp_folder'])

    data_folder = path.join(DATA_DIR, "raw", dataset_name)
    if not sub_folder is None:
        data_folder = path.join(data_folder, sub_folder)

    split_dir = path.join(DATA_DIR, "splits",
                          "{}-{}".format(dataset_name, split_name))
    train_meta = path.join(split_dir, "train.json")
    test_meta = path.join(split_dir, "test.json")
    validation_meta = None
    if not __is_kfold(data_config):
        validation_meta = path.join(split_dir, "val.json")

    ret = dict(
        data_folder=data_folder,
        train_meta=train_meta,
        validation_meta=validation_meta,
        test_meta=test_meta,
        temp_folder=temp_folder,
        **data_params
    )

    return ret


def __make_datasets(DataClass, data_folder, train_meta, validation_meta=None, test_meta=None, temp_folder=None, force_compute=False, sr=22050, duration=5.0, overlap=2.5, ext="mp3"):
    train_ds = DataClass(train_meta, data_folder, temp_folder=temp_folder, chunk_duration=duration,
                         overlap=overlap, force_compute=force_compute, sr=sr, audio_extension=ext)
    test_ds = DataClass(test_meta, data_folder, temp_folder=temp_folder, chunk_duration=duration,
                        overlap=overlap, force_compute=force_compute, sr=sr, audio_extension=ext)
    validation_ds = None
    if not validation_meta is None:
        validation_ds = DataClass(train_meta, data_folder, temp_folder=temp_folder, chunk_duration=duration,
                                  overlap=overlap, force_compute=force_compute, sr=sr, audio_extension=ext)
    return (train_ds, test_ds, validation_ds)

def parse_model_args(args):
    ret = {}
    for (i, arg) in enumerate(args):
        if arg.startswith("--"):
            n = arg[2:].replace("-", "_")
            a = n.split("=")
            if len(a) == 1:
                # parse parameter
                if len(args) > i+1 and not args[i+1].startswith("--"):
                    v = args[i+1]
                    try:
                        v = int(v)
                    except:
                        pass
                    ret[n] = v
                else:
                    ret[n] = True
            elif len(a) == 2:
                v = a[1]
                try:
                    v = int(v)
                except:
                    pass
                ret[a[0]] = v
    return ret


WORKING_DIR = path.dirname(__file__)
BASE_CONFIG = __load_default_config()
DATA_DIR = BASE_CONFIG['data_dir']
TEMP_DIR = BASE_CONFIG['temp_dir']


@click.group()
def cli():
    pass


@click.group("list")
def clist():
    pass


@click.command("runs")
@click.argument("subpath", required=False)
@click.option('--print/--no-print', 'print_file', default=False, help="Print contents of the runs.")
def list_runs(subpath, print_file):
    if subpath:
        sp = subpath.replace(".", "/")
        runs_dir = path.join(WORKING_DIR, "runs", sp)
    else:
        runs_dir = path.join(WORKING_DIR, "runs")
    found = 0
    for (c_dir, _, files) in walk(runs_dir):
        if "__pycache__" in c_dir:
            continue
        files = list(filter(lambda x: x.endswith(".yaml"), files))
        if len(files) == 0:
            continue
        p = c_dir[len("runs/"):].replace("/", ".")
        for f in files:
            if f.endswith(".yaml"):
                print("{}.{}".format(p, f[:-len(".yaml")]))
                if print_file:
                    print("-"*20)
                    print("".join(open(path.join(c_dir, f), mode="r").readlines()))
                found += 1
        print("="*20)
    print("Runs Found: {}".format(found))


@click.command("models")
@click.argument("subpath", required=False)
def list_models(subpath):
    if subpath:
        sp = subpath.replace(".", "/")
        models_dir = path.join(WORKING_DIR, "models", sp)
    else:
        models_dir = path.join(WORKING_DIR, "models")
    found = 0
    for (c_dir, _, files) in walk(models_dir):
        if "__pycache__" in c_dir:
            continue
        files = list(filter(lambda x: x.startswith("model_v"), files))
        if len(files) == 0:
            continue
        p = c_dir[len("models/"):].replace("/", ".")
        print("{}".format(p))
        print("-"*20)
        found += 1
        for f in files:
            model_info = __get_model_info(path.join(c_dir, f))
            print("{}:{}".format(model_info['name'], model_info['version']))
        print("="*20)
    print("Models Found: {}".format(found))


@click.command("check")
@click.argument("run")
@click.option("--data/--no-data", "check_data", default=True, help="Load dataset and run forward pass.")
@click.option("--model-version", type=int, required=False)
def check(run, check_data, model_version):
    run_dir = path.join(WORKING_DIR, "runs")
    (rd, run_file) = __parse_run_location(run)
    run_dir = path.join(run_dir, rd)

    run_config = __load_yaml_file(path.join(run_dir, run_file))
    if not model_version is None:
        run_config['model']['version'] = model_version
        
    (ModelClass, _) = __load_model_class(run, run_config['model']['version'])

    model_params = run_config['model']['params']

    model = ModelClass(**model_params)
    print("Model Created...")

    data_args = __parse_data_args(run_config['data'])
    print("DataClass Args:")
    print("Data Folder: {}".format(data_args['data_folder']))
    print("Temp Folder: {}".format(data_args['temp_folder']))

    if check_data:
        DataClass = __load_data_class(run)

        dss = __make_datasets(DataClass, **data_args)
        for ds in dss:
            if ds is None:
                continue
            dl = DataLoader(ds, batch_size=2, num_workers=2, drop_last=True)
            for (X, _) in dl:
                model(X)
                break
        print("Check: forward passes ok!")

    sr = data_args['sr'] if 'sr' in data_args else 22050
    duration = data_args['duration'] if 'duration' in data_args else 5.0

    print(torchinfo.summary(model, input_size=(2, 1, int(sr * duration))))

@click.command("train", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("run", required=True)
@click.option("--wandb/--no-wandb", "use_wandb", default=True, help="Use WandB to log metrics")
@click.option("--batch-size", type=int, required=False)
@click.option("--temp-folder", type=str, required=False)
@click.option("--model-version", type=int, required=False)
@click.option("--dataset", type=str, required=False)
@click.option("--split", type=str, required=False)
@click.pass_context
def train(ctx: click.Context, run, use_wandb, batch_size, temp_folder, model_version, dataset, split):

    run_dir = path.join(WORKING_DIR, "runs")

    (rd, run_file) = __parse_run_location(run)
    run_dir = path.join(run_dir, rd)

    run_config = __load_yaml_file(path.join(run_dir, run_file))

    if not batch_size is None:
        run_config['batch_size'] = batch_size
    if not temp_folder is None:
        run_config['data']['temp_folder'] = temp_folder
    if not dataset is None:
        run_config['data']['dataset'] = dataset
    if not split is None:
        run_config['data']['split'] = split
    if not model_version is None:
        run_config['model']['version'] = model_version

    batch_size = run_config['batch_size']
    
    model_args_additional = parse_model_args(ctx.args)

    (ModelClass, model_info) = __load_model_class(
        run, run_config['model']['version'])
    DataClass = __load_data_class(run)

    data_args = __parse_data_args(run_config['data'])
    (train_ds, test_ds, validation_ds) = __make_datasets(DataClass, **data_args)
    print("Datasets Created...")

    model_params = {
        **run_config['model']['params'],
        **model_args_additional
    }

    additional_tags = run_config['tags'] if 'tags' in run_config else []

    config = {
        **model_params
    }

    if __is_kfold(run_config['data']):
        kfold_n = run_config['kfold_n'] if 'kfold_n' in run_config else 5
        stratify = run_config['stratify'] if 'stratify' in run_config else True


        model = ModelClass(**model_params)
        print("Model Created...")

        cv = kfold.CrossValidator(
            n_splits=kfold_n,
            stratify=stratify,
            batch_size=batch_size,
            num_workers=__get_num_workers(),
            wandb_project_name="mer",
            model_monitor=model.MODEL_CHECKPOINT,
            model_monitor_mode=model.MODEL_CHECKPOINT_MODE,
            early_stop_monitor=model.EARLY_STOPPING,
            early_stop_mode=model.EARLY_STOPPING_MODE,
            use_wandb=use_wandb,
            cv_dry_run=False,
            wandb_tags=__get_wandb_tags(
                model_info['name'], model_info['version'], run_config['data']['dataset'], additional_tags),
            config=config,
            gpus=__get_gpu_count()
        )

        cv.fit(model, train_ds, test_ds)

        return
    
    model = ModelClass(train_ds=train_ds, test_ds=test_ds, val_ds=validation_ds, batch_size=batch_size, **model_params)
    print("Model Created...")

    model_callback = ModelCheckpoint(
        monitor=model.MODEL_CHECKPOINT, mode=model.MODEL_CHECKPOINT_MODE)
    early_stop_callback = EarlyStopping(
        monitor=model.EARLY_STOPPING,
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode=model.EARLY_STOPPING_MODE
    )

    logger = None
    if use_wandb:
        logger = WandbLogger(
            offline=False,
            log_model=True,
            project='mer',
            job_type="train",
            config=config,
            tags=__get_wandb_tags(
                model_info['name'], model_info['version'], run_config['data']['dataset'], additional_tags)
        )

    trainer = pl.Trainer(
        logger=logger,
        gpus=__get_gpu_count(),
        callbacks=[model_callback, early_stop_callback])

    trainer.fit(model)

    trainer.test(model)

    if use_wandb:
        wandb.finish()

@click.command("sweep", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("run", required=True)
@click.option("--batch-size", type=int, required=False, default=lambda: os.environ.get("BATCH_SIZE", None))
@click.option("--dataset", type=str, required=False)
@click.option("--split", type=str, required=False)
@click.option("--temp-folder", type=str, required=False)
@click.option("--model-version", type=int, required=False)
@click.option("--auto-batch-size/--no-auto-batch-size", default=False)
def sweep(run, batch_size, dataset, split, temp_folder, model_version, auto_batch_size):

    run_dir = path.join(WORKING_DIR, "runs")

    (rd, run_file) = __parse_run_location(run)
    run_dir = path.join(run_dir, rd)

    run_config = __load_yaml_file(path.join(run_dir, run_file))

    if not batch_size is None:
        run_config['batch_size'] = batch_size
    if not temp_folder is None:
        run_config['data']['temp_folder'] = temp_folder
    if not dataset is None:
        run_config['data']['dataset'] = dataset
    if not split is None:
        run_config['data']['split'] = split
    if not model_version is None:
        run_config['model']['version'] = model_version

    (ModelClass, model_info) = __load_model_class(run, run_config['model']['version'])
    DataClass = __load_data_class(run)

    data_args = __parse_data_args(run_config['data'])
    (train_ds, test_ds, validation_ds) = __make_datasets(DataClass, **data_args)
    print(f"Datasets {run_config['data']['dataset']} Created...")
    print(f"Using Temp Folder - {run_config['data']['temp_folder']}")
    print(f"Split name - {run_config['data']['split']}")

    default_model_params = run_config['model']['params']
    batch_size = run_config['batch_size']

    # Pass your defaults to wandb.init
    wandb_exp = wandb.init(config=default_model_params)

    # Access all hyperparameter values through wandb.config
    config = wandb.config

    model = ModelClass(train_ds=train_ds, test_ds=test_ds, val_ds=validation_ds, batch_size=batch_size, **config)
    print("Model Created...")

    additional_tags = run_config['tags'] if 'tags' in run_config else []

    model_callback = ModelCheckpoint(
        monitor=model.MODEL_CHECKPOINT, mode=model.MODEL_CHECKPOINT_MODE)
    early_stop_callback = EarlyStopping(
        monitor=model.EARLY_STOPPING,
        min_delta=0.001,
        patience=15,
        verbose=True,
        mode=model.EARLY_STOPPING_MODE
    )

    logger = WandbLogger(
        experiment=wandb_exp,
        offline=False,
        log_model=True,
        job_type="train",
        config=config,
        tags=__get_wandb_tags(
            model_info['name'], model_info['version'], run_config['data']['dataset'], additional_tags)
    )

    trainer = pl.Trainer(
        logger=logger,
        gpus=__get_gpu_count(),
        callbacks=[model_callback, early_stop_callback],
        auto_scale_batch_size=auto_batch_size)

    if auto_batch_size:
        trainer.tune(model)

    trainer.fit(model)

    trainer.test(model)



clist.add_command(list_runs)
clist.add_command(list_models)

cli.add_command(clist)
cli.add_command(check)
cli.add_command(train)
cli.add_command(sweep)

if __name__ == "__main__":
    cli()
