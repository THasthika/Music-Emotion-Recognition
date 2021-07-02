from re import sub
from dotenv import load_dotenv
import argparse
import multiprocessing
from torch.utils.data.dataloader import DataLoader
import torchinfo

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torch.cuda

from model_v1 import A2DConvCat_V1 as ModelV1
from model_v2 import A2DConvCat_V2 as ModelV2
from model_v3 import A2DConvCat_V3 as ModelV3
from kfold import CrossValidator
from data import ModelDataset

load_dotenv(verbose=True)


def make_datasets(args):
    data_folder = args['data_folder']
    train_meta = args['train_meta']
    validation_meta = args['validation_meta']
    test_meta = args['test_meta']
    temp_folder = args['temp_folder']
    force_compute = args['force_compute']

    sr = args['sr']
    duration = args['duration']
    overlap = args['overlap']
    ext = args['ext']

    train_ds = ModelDataset(train_meta, data_folder, temp_folder=temp_folder, chunk_duration=duration,
                            overlap=overlap, force_compute=force_compute, sr=sr, audio_extension=ext)
    test_ds = ModelDataset(test_meta, data_folder, temp_folder=temp_folder, chunk_duration=duration,
                           overlap=overlap, force_compute=force_compute, sr=sr, audio_extension=ext)
    validation_ds = None
    if not validation_meta is None:
        validation_ds = ModelDataset(train_meta, data_folder, temp_folder=temp_folder, chunk_duration=duration,
                                     overlap=overlap, force_compute=force_compute, sr=sr, audio_extension=ext)
    return (train_ds, test_ds, validation_ds)


def make_model(args, train_ds, test_ds, validation_ds=None):
    batch_size = args['batch_size']
    num_workers = args['num_workers']

    model_version = args['model_version']

    if model_version == 1:
        args_k = [ModelV1.LR, ModelV1.N_FFT, ModelV1.ADAPTIVE_LAYER_UNITS_0,
                  ModelV1.ADAPTIVE_LAYER_UNITS_1, ModelV1.SPEC_TRAINABLE]
        model_config = {k: args[k] for k in args_k}
        model = ModelV1(batch_size=batch_size, num_workers=num_workers,
                        train_ds=train_ds, val_ds=validation_ds, test_ds=test_ds, **model_config)
        return (model, model_config)
    elif model_version == 2:
        args_k = [ModelV2.LR, ModelV2.N_FFT, ModelV2.ADAPTIVE_LAYER_UNITS_0,
                  ModelV2.ADAPTIVE_LAYER_UNITS_1, ModelV2.SPEC_TRAINABLE, ModelV2.N_MELS]
        model_config = {k: args[k] for k in args_k}
        model = ModelV2(batch_size=batch_size, num_workers=num_workers,
                        train_ds=train_ds, val_ds=validation_ds, test_ds=test_ds, **model_config)
        return (model, model_config)
    elif model_version == 3:
        args_k = [ModelV3.LR, ModelV3.N_FFT, ModelV3.ADAPTIVE_LAYER_UNITS_0,
                  ModelV3.ADAPTIVE_LAYER_UNITS_1, ModelV3.SPEC_TRAINABLE, ModelV3.N_MELS, ModelV3.N_MFCC]
        model_config = {k: args[k] for k in args_k}
        model = ModelV3(batch_size=batch_size, num_workers=num_workers,
                        train_ds=train_ds, val_ds=validation_ds, test_ds=test_ds, **model_config)
        return (model, model_config)
    else:
        raise ModuleNotFoundError()


def get_gpu_count():
    if torch.cuda.is_available():
        return -1
    return None


def get_num_workers():
    return multiprocessing.cpu_count()


def get_wandb_tags(args):
    return [
        'model:A2DConvCat',
        'dataset:{}'.format(args['dataset']),
        'version:{}'.format(args['model_version'])
    ]


def train_kfold(args):
    (train_ds, test_ds, _) = make_datasets(args)
    (model, model_config) = make_model(args, None, None)

    config = {
        **model_config
    }

    cv = CrossValidator(
        n_splits=args['kfold_k'],
        stratify=False,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        wandb_project_name="mer",
        model_monitor=model.MODEL_CHECKPOINT,
        model_monitor_mode=model.MODEL_CHECKPOINT_MODE,
        early_stop_monitor=model.EARLY_STOPPING,
        early_stop_mode=model.EARLY_STOPPING_MODE,
        use_wandb=(not args['no_wandb']),
        cv_dry_run=False,
        wandb_tags=get_wandb_tags(args),
        config=config,
        gpus=get_gpu_count()
    )

    cv.fit(model, train_ds, test_ds)


def check(args):

    (train_ds, test_ds, validation_ds) = make_datasets(args)
    (model, _) = make_model(args, train_ds, test_ds, validation_ds)

    for ds in [train_ds, test_ds, validation_ds]:
        if ds is None:
            continue

        dl = DataLoader(ds, batch_size=2, num_workers=2, drop_last=True)

        for (X, _) in dl:
            model(X)
            break

    print("Model: foward passes ok!")

    print(torchinfo.summary(model, input_size=(2, 1, 22050*5)))


def train(args):

    if args['check']:
        check(args)
        return

    if args['kfold']:
        train_kfold(args)
        return

    (train_ds, test_ds, validation_ds) = make_datasets(args)
    (model, model_config) = make_model(args, train_ds, test_ds, validation_ds)

    config = {
        **model_config
    }

    model_callback = ModelCheckpoint(
        monitor=model.MODEL_CHECKPOINT, mode=model.MODEL_CHECKPOINT_MODE)
    early_stop_callback = EarlyStopping(
        monitor=model.EARLY_STOPPING,
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode=model.EARLY_STOPPING_MODE
    )

    logger = None
    if not args['no_wandb']:
        logger = WandbLogger(
            offline=False,
            log_model=True,
            project='mer',
            job_type="train",
            config=config,
            tags=get_wandb_tags(args)
        )

    trainer = pl.Trainer(
        logger=logger,
        gpus=get_gpu_count(),
        callbacks=[model_callback, early_stop_callback])

    trainer.fit(model)

    trainer.test(model)


def main(in_args=None):
    parser = argparse.ArgumentParser(prog="Model")

    subparsers = parser.add_subparsers(help='sub programs')
    subparser_train = subparsers.add_parser('train', help='train the model')
    subparser_train.set_defaults(func=train)
    subparser_train.add_argument(
        '--num-workers', type=int, default=get_num_workers())
    subparser_train.add_argument(
        '--no-wandb', action='store_true', default=False)
    subparser_train.add_argument('--kfold', action='store_true', default=False)
    subparser_train.add_argument('--kfold-k', type=int, default=5)
    subparser_train.add_argument('--model-version', type=int, default=1)

    model_args = subparser_train.add_argument_group('Model Arguments')
    model_args.add_argument('--check', action='store_true', default=False)
    model_args.add_argument('--lr', '--learning-rate',
                            type=float, default=0.01)
    model_args.add_argument('--n-fft', type=int, default=2048)
    model_args.add_argument('--n-mels', type=int, default=128)
    model_args.add_argument('--n-mfcc', type=int, default=20)
    model_args.add_argument('--adaptive-layer-units-0',
                            type=int, default=128)
    model_args.add_argument('--adaptive-layer-units-1',
                            type=int, default=128)
    model_args.add_argument('--batch-size', type=int, default=32)
    model_args.add_argument(
        '--spec-trainable', action='store_true', default=False)

    data_args = subparser_train.add_argument_group('Dataset Arguments')
    data_args.add_argument('--dataset', type=str, required=True)
    data_args.add_argument('--data-folder', type=str, required=True)
    data_args.add_argument('--train-meta', type=str, required=True)
    data_args.add_argument('--validation-meta', type=str,
                           required=False, default=None)
    data_args.add_argument('--test-meta', type=str, required=True)
    data_args.add_argument('--temp-folder', type=str, required=True)
    data_args.add_argument(
        '--force-compute', action='store_true', default=False)

    audio_args = subparser_train.add_argument_group('Audio Arguments')
    audio_args.add_argument('--sr', '--sample-rate', type=int, default=22050)
    audio_args.add_argument('--duration', type=float, default=5)
    audio_args.add_argument('--overlap', type=float, default=2.5)
    audio_args.add_argument('--ext', '--extention', type=str, default='mp3')

    args = parser.parse_args(in_args)
    args.func(vars(args))


if __name__ == "__main__":
    main()
