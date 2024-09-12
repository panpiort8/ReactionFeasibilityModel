import gin
import torchmetrics

gin.external_configurable(torchmetrics.Accuracy, module="tm")
gin.external_configurable(torchmetrics.AUROC, module="tm")
gin.external_configurable(torchmetrics.Precision, module="tm")
gin.external_configurable(torchmetrics.F1Score, module="tm")
gin.external_configurable(torchmetrics.Recall, module="tm")
gin.external_configurable(torchmetrics.AveragePrecision, module="tm")
