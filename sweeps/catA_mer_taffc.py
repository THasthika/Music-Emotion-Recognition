import wandb

import pytorch_lightning as pl

from models import ModelCatA

CMDS = ModelCatA.CMDS

DEFAULT_DATA_ARTIFACT = "mer-taffc:latest"
DEFAULT_SPLIT_ARTIFACT = "mer-taffc-train-70-val-20-test-10-seed-42:latest"
DEFAULT_CONFIG = dict(map(lambda x: (x[0], x[2]), CMDS))

def makeTrainer(project_name, data_artifact, split_artifact, batch_size, find_batch_size, find_lr):

    def train(config=None):
        if config is None:
            config = DEFAULT_CONFIG

        run = wandb.init(config=config, project=project_name, job_type="train")

        config = dict(run.config)

        print("config:", config)

        logger = WandbLogger(project=project_name, experiment=run, log_model=True)

        model = ModelCatA(data_artifact=data_artifact, split_artifact=split_artifact, batch_size=batch_size, **config)

        checkpoint_callback = ModelCheckpoint(monitor='val/acc')

        trainer = pl.Trainer(
            logger=logger,
            gpus=-1,
            checkpoint_callback=True,
            progress_bar_refresh_rate=1,
            max_epochs=config['max_epochs'],
            auto_scale_batch_size=find_batch_size,
            )

        if tune:
            trainer.tune(model)

        trainer.fit(model)

        if not model.test_ds is None:
            trainer.test(model, ckpt_path='best')
    
    return train

def get_parse_args():
    return (DEFAULT_DATA_ARTIFACT, DEFAULT_SPLIT_ARTIFACT, CMDS)


