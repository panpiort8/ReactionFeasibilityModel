from typing import Dict

import dgl
import gin
import pytorch_lightning as pl
import torchmetrics
from models import ReactionGNN
from torch import nn, optim
from torchmetrics import Metric


@gin.configurable()
class ReactionGNNModule(pl.LightningModule):
    def __init__(
        self,
        model: ReactionGNN,
        lr: float,
        train_metrics: Dict[str, Metric],
        valid_metrics: Dict[str, Metric],
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.valid_metrics = nn.ModuleDict(valid_metrics)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def training_step(self, batch: dgl.DGLGraph, batch_idx: int):
        reactants, products, labels = batch
        logits = self.model(reactants, products)
        loss = self.loss_fn(logits, labels)

        for metric in self.train_metrics.values():
            metric.update(logits, labels.long())
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        for name, metric in self.train_metrics.items():
            self.log(f"train/{name}", metric, on_epoch=True, on_step=False)

    def validation_step(self, batch: dgl.DGLGraph, batch_idx: int):
        reactants, products, labels = batch
        logits = self.model(reactants, products)
        loss = self.loss_fn(logits, labels)

        for metric in self.valid_metrics.values():
            metric.update(logits, labels.long())

        return loss

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.valid_metrics.items():
            self.log(f"valid/{name}", metric, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
