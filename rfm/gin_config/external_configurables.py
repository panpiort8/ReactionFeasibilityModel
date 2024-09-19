import gin
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

gin.external_configurable(torchmetrics.Accuracy, module="tm")
gin.external_configurable(torchmetrics.AUROC, module="tm")
gin.external_configurable(torchmetrics.Precision, module="tm")
gin.external_configurable(torchmetrics.F1Score, module="tm")
gin.external_configurable(torchmetrics.Recall, module="tm")
gin.external_configurable(torchmetrics.AveragePrecision, module="tm")
gin.external_configurable(DataLoader, module="tm")
gin.external_configurable(WandbLogger, module="wandb")
gin.external_configurable(ModelCheckpoint, module="pl")
gin.external_configurable(pl.Trainer, module="pl")
