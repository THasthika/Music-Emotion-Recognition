from sklearn.model_selection import KFold, StratifiedKFold

import pytorch_lightning as pl

from torch.utils.data import Dataset, Subset, DataLoader

from coolname import generate_slug

class KFoldHelper:
    """Split data for (Stratified) K-Fold Cross-Validation."""
    def __init__(self,
                 n_splits=5,
                 stratify=False,
                 batch_size=16,
                 num_workers=4):
        super().__init__()
        self.n_splits = n_splits
        self.stratify = stratify
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, data):
        if self.stratify:
            labels = data.get_labels()
            splitter = StratifiedKFold(n_splits=self.n_splits)
        else:
            labels = None
            splitter = KFold(n_splits=self.n_splits)

        n_samples = len(data)
        for train_idx, val_idx in splitter.split(X=range(n_samples), y=labels):

            train_dataset = Subset(data, train_idx)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers)

            val_dataset = Subset(data, val_idx)
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers)

            yield train_loader, val_loader

class WandBCV:
    """Cross-validation with a LightningModule."""
    def __init__(self,
                 n_splits=5,
                 stratify=False,
                 batch_size=16,
                 num_workers=4,
                 wandb_group=None,
                 wandb_project_name=None,
                 wandb_tags=None,
                 *trainer_args,
                 **trainer_kwargs):
        super().__init__()
        self.trainer_args = trainer_args
        self.trainer_kwargs = trainer_kwargs
        self.n_splits = n_splits
        self.stratify = stratify
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wandb_tags = wandb_tags
        self.wandb_group = wandb_group
        self.wandb_project_name = wandb_project_name

        self.run_name = generate_slug(2)

    def fit(self, model: pl.LightningModule, data: Dataset):
        split_func = KFoldHelper(
            n_splits=self.n_splits,
            stratify=self.stratify,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
        cv_data = split_func(data)
        for fold_idx, loaders in enumerate(cv_data):

            # Clone model & instantiate a new trainer:
            _model = deepcopy(model)
            logger = WandbLogger(
                offline=False,
                project=self.wandb_project_name,
                group=self.wandb_group,
                job_type="train",
                tags=self.wandb_tags,
                name="{}-fold-{}".format(self.run_name, fold_idx)
            )
            trainer = pl.Trainer(logger=logger, *self.trainer_args, **self.trainer_kwargs)

            # Update loggers and callbacks:
            self.update_logger(trainer, fold_idx)
            for callback in trainer.callbacks:
                if isinstance(callback, pl.callbacks.ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, *loaders)

            wandb.finish()