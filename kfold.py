from sklearn.model_selection import KFold, StratifiedKFold
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from coolname import generate_slug
from copy import deepcopy
import wandb
import os

from yaml import load

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
                                      drop_last=True,
                                      num_workers=self.num_workers)

            val_dataset = Subset(data, val_idx)
            val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    drop_last=True,
                                    num_workers=self.num_workers)

            yield train_loader, val_loader

class CrossValidator:
    """Cross-validation with a LightningModule."""
    def __init__(self,
                 n_splits=5,
                 stratify=False,
                 batch_size=16,
                 num_workers=4,
                 max_runs=None,
                 wandb_group=None,
                 wandb_project_name=None,
                 wandb_tags=None,
                 model_monitor='val/loss',
                 model_monitor_mode='min',
                 early_stop_monitor='val/acc',
                 early_stop_mode='max',
                 config={},
                 use_wandb=True,
                 cv_dry_run=False,
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
        self.wandb_project_name = wandb_project_name
        self.max_runs = max_runs

        self.run_name = generate_slug(2)
        if wandb_group is None:
            wandb_group = self.run_name
        self.wandb_group = wandb_group

        self.model_monitor = model_monitor
        self.model_monitor_mode = model_monitor_mode
        self.early_stop_monitor = early_stop_monitor
        self.early_stop_mode = early_stop_mode

        self.cv_dry_run = cv_dry_run
        if cv_dry_run:
            print("Warning: CV - Dry Run")
        
        self.use_wandb = use_wandb
        if 'USE_WANDB' in os.environ.keys():
            self.use_wandb = (False if os.environ.get('USE_WANDB', 'true').lower() == 'false' else True)

        if not self.use_wandb:
            print("Not Using WandB")

    def fit(self, model: pl.LightningModule, data: Dataset, test_data: Dataset):
        # model = model.to('cpu')

        print("Initiating KFoldHelper...")
        split_func = KFoldHelper(
            n_splits=self.n_splits,
            stratify=self.stratify,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
        test_dl = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.num_workers)

        print("KFoldHelper: Splitting dataset into {} folds".format(self.n_splits))
        cv_data = split_func(data)
        for fold_idx, loaders in enumerate(cv_data):

            if not self.max_runs is None and self.max_runs < fold_idx:
                print("Reached Maximum Runs... {}".format(self.max_runs))
                break

            print("Starting {} Fold...".format(fold_idx))

            train_dl, val_dl = loaders

            # accommodate dry run to check model is working
            if self.cv_dry_run:
                for (dl_i, dl) in enumerate([train_dl, val_dl, test_dl]):
                    print("Checking DataLoader - {}".format(dl_i))
                    for (X, y) in dl:
                        print("Input Dimensions")
                        if type(X) is dict:
                            for k in X:
                                print("{} - {}".format(k, X[k].shape))
                        elif type(X) is list:
                            for (i, x) in enumerate(X):
                                print("{} - {}".format(i, x.shape))
                        else:
                            print(X.shape)
                        print("Target Dimensions")
                        print(y.shape)
                        break
                continue

            # Clone model & instantiate a new trainer:
            _model = deepcopy(model)
            logger = None
            if self.use_wandb:
                logger = WandbLogger(
                    offline=False,
                    log_model=True,
                    project=self.wandb_project_name,
                    group=self.wandb_group,
                    job_type="train",
                    tags=self.wandb_tags,
                    name="{}-fold-{}".format(self.run_name, fold_idx)
                )

            model_callback = ModelCheckpoint(
                monitor=self.model_monitor,
                mode=self.model_monitor_mode
            )
            early_stop_callback = EarlyStopping(
                monitor=self.early_stop_monitor,
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode=self.early_stop_mode
            )

            trainer = pl.Trainer(
                logger=logger,
                callbacks=[model_callback, early_stop_callback],
                *self.trainer_args,
                **self.trainer_kwargs)

            # # Update loggers and callbacks:
            # self.update_logger(trainer, fold_idx)
            # for callback in trainer.callbacks:
            #     if isinstance(callback, pl.callbacks.ModelCheckpoint):
            #         self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, train_dataloader=train_dl, val_dataloaders=val_dl)

            trainer.test(_model, test_dl)

            if self.use_wandb:
                wandb.finish()