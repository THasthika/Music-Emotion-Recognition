from models.base import BaseModel

from models.base import BaseModel
import torchmetrics as tm
import torch.nn as nn

from utils.activation import CustomELU
from utils.loss import rmse_loss

class BaseStatModel(BaseModel):

    STD_ACTIVATION = "std_activation"

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)

        ## loss
        self.loss = rmse_loss
        
        self.test_arousal_mean_r2 = tm.R2Score(num_outputs=1)
        self.test_valence_mean_r2 = tm.R2Score(num_outputs=1)
        self.test_arousal_std_r2 = tm.R2Score(num_outputs=1)
        self.test_valence_std_r2 = tm.R2Score(num_outputs=1)

        self.test_mean_r2score = tm.R2Score(num_outputs=2)

    def _get_std_activation(self):
        std_activation = None
        if self.config[self.STD_ACTIVATION] == "custom":
            print("Model: StdActivation uses CustomELU")
            std_activation = CustomELU(alpha=1.0)
        elif self.config[self.STD_ACTIVATION] == "relu":
            print("Model: StdActivation uses ReLU")
            std_activation = nn.ReLU()
        elif self.config[self.STD_ACTIVATION] == "softplus":
            print("Model: StdActivation uses Softplus")
            std_activation = nn.Softplus()
        if std_activation is None:
            raise Exception("Activation Type Unknown!")
        return std_activation
    
    def predict(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        arousal_std_rmse = self.loss(pred[:, 3], y[:, 3])
        valence_std_rmse = self.loss(pred[:, 2], y[:, 2])

        arousal_mean_rmse = self.loss(pred[:, 1], y[:, 1])
        valence_mean_rmse = self.loss(pred[:, 0], y[:, 0])

        self.log("val/loss", loss, prog_bar=True)

        self.log('val/arousal_std_rmse', arousal_std_rmse, on_step=False, on_epoch=True)
        self.log('val/valence_std_rmse', valence_std_rmse, on_step=False, on_epoch=True)

        self.log("val/arousal_mean_rmse", arousal_mean_rmse, on_step=False, on_epoch=True)
        self.log("val/valence_mean_rmse", valence_mean_rmse, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        arousal_std_rmse = self.loss(pred[:, 3], y[:, 3])
        valence_std_rmse = self.loss(pred[:, 2], y[:, 2])

        arousal_mean_rmse = self.loss(pred[:, 1], y[:, 1])
        valence_mean_rmse = self.loss(pred[:, 0], y[:, 0])

        mean_r2score = self.test_mean_r2score(pred[:, [0, 1]], y[:, [0, 1]])

        arousal_mean_r2score = self.test_arousal_mean_r2(pred[:, 1], y[:, 1])
        valence_mean_r2score = self.test_valence_mean_r2(pred[:, 0], y[:, 0])

        arousal_std_r2score = self.test_arousal_std_r2(pred[:, 3], y[:, 3])
        valence_std_r2score = self.test_valence_std_r2(pred[:, 2], y[:, 2])

        self.log("test/loss", loss)

        self.log('test/mean_r2score', mean_r2score, on_step=False, on_epoch=True)

        self.log('test/arousal_mean_r2score', arousal_mean_r2score, on_step=False, on_epoch=True)
        self.log('test/valence_mean_r2score', valence_mean_r2score, on_step=False, on_epoch=True)

        self.log('test/arousal_std_r2score', arousal_std_r2score, on_step=False, on_epoch=True)
        self.log('test/valence_std_r2score', valence_std_r2score, on_step=False, on_epoch=True)

        self.log("test/arousal_mean_rmse", arousal_mean_rmse, on_step=False, on_epoch=True)
        self.log("test/valence_mean_rmse", valence_mean_rmse, on_step=False, on_epoch=True)

        self.log('val/arousal_std_rmse', arousal_std_rmse, on_step=False, on_epoch=True)
        self.log('val/valence_std_rmse', valence_std_rmse, on_step=False, on_epoch=True)