from models.base import BaseModel
import torchmetrics as tm
import torch
import torch.nn.functional as F

class BaseCatModel(BaseModel):

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)

        ## metrics
        self.train_acc = tm.Accuracy(top_k=3)

        self.val_acc = tm.Accuracy(top_k=3)
        self.val_f1_class = tm.F1(num_classes=4, average='none')
        self.val_f1_global = tm.F1(num_classes=4)

        self.test_acc = tm.Accuracy(top_k=3)
        self.test_f1_class = tm.F1(num_classes=4, average='none')
        self.test_f1_global = tm.F1(num_classes=4)
        self.test_auroc = tm.AUROC(num_classes=4, compute_on_step=False)

        ## loss
        self.loss = F.cross_entropy
    
    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        pred = F.softmax(y_logit, dim=1)

        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/acc', self.train_acc(pred, y), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        pred = F.softmax(y_logit, dim=1)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True)

        self.log("val/f1_global", self.val_f1_global(pred, y), on_step=False, on_epoch=True)

        f1_scores = self.val_f1_class(pred, y)
        for (i, x) in enumerate(torch.flatten(f1_scores)):
            self.log("val/f1_class_{}".format(i), x, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self(x)
        loss = self.loss(y_logit, y)
        pred = F.softmax(y_logit, dim=1)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc(pred, y))
        self.log("test/f1_global", self.test_f1_global(pred, y))

        self.test_auroc(pred, y)

        f1_scores = self.test_f1_class(pred, y)
        for (i, x) in enumerate(torch.flatten(f1_scores)):
            self.log("test/f1_class_{}".format(i), x)

    def test_step_end(self, output_results):
        self.log("test/auroc", self.test_auroc.compute(), on_step=False)