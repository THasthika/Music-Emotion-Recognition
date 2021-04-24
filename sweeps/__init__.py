import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def makeSweepTrainer(modelClass, default_config=None, gpus=-1, monitor='val/loss'):

    def makeTrainer(project_name, data_artifact, split_artifact, batch_size, find_batch_size, find_lr):

        def train(config=None):

            if config is None:
                config = default_config

            run = wandb.init(config=config, project=project_name, job_type="train")

            config = dict(run.config)

            print("config:", config)

            logger = WandbLogger(project=project_name, experiment=run, log_model=True, offline=False)

            model = modelClass(data_artifact=data_artifact, split_artifact=split_artifact, batch_size=batch_size, **config)

            checkpoint_callback = ModelCheckpoint(monitor=monitor)

            trainer = pl.Trainer(
                logger=logger,
                gpus=gpus,
                checkpoint_callback=checkpoint_callback,
                progress_bar_refresh_rate=1,
                max_epochs=config['max_epochs'],
                auto_scale_batch_size=find_batch_size,
                )

            if find_batch_size or find_lr:
                trainer.tune(model)

            trainer.fit(model)

            if not model.test_ds is None:
                trainer.test(model, ckpt_path='best')
        
        return train

    return makeTrainer