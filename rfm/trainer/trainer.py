from pathlib import Path
from typing import Dict, Literal, Optional

import gin
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from .logger.logger_base import LoggerBase
from .optimizer_base import OptimizerBase
from .reaction_dataset import ReactionDataset
from .utils import infer_metric_direction


@gin.configurable()
class ReactionTrainer:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        train_dataset: ReactionDataset,
        train_batch_size: int,
        valid_dataset: ReactionDataset,
        valid_batch_size: int,
        train_metrics: Dict[str, Metric],
        valid_metrics: Dict[str, Metric],
        model: nn.Module,
        logger: LoggerBase,
        optimizer: OptimizerBase,
        n_epochs: int,
        device: str = "auto",
        checkpoint_best: bool = False,
        best_metric: str = "loss",
        metric_direction: Literal["auto", "min", "max"] = "auto",
        gradient_clipping_norm: float = 10.0,
        num_workers: int = 0,
        valid_every_n_epochs: int = 5,
        log_train_every_n_batches: int = 10,
    ):
        assert metric_direction in ("auto", "min", "max")
        self.run_dir = Path(run_dir)
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics
        self.logger = logger
        self.optimizer = optimizer
        self.model = model
        self.n_epochs = n_epochs
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint_best = checkpoint_best
        self.best_metric = best_metric
        self.gradient_clipping_norm = gradient_clipping_norm
        self.valid_every_n_epochs = valid_every_n_epochs
        self.log_train_every_n_batches = log_train_every_n_batches

        self.metric_direction = (
            infer_metric_direction(self.best_metric)
            if metric_direction == "auto"
            else metric_direction
        )
        self.best_valid_metrics: Dict[str, float] = {}
        # if not train_dataset.is_preprocessed():
        #     train_dataset.preprocess()
        # if not valid_dataset.is_preprocessed():
        #     valid_dataset.preprocess()

        self.model.to(device)
        self.optimizer.initialize(model=model)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_dataset.collate,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=valid_dataset.collate,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def valid_step(self) -> Dict[str, float]:
        self.model.eval()
        losses = []
        logits_list = []
        for reactants, products, labels in tqdm(self.valid_loader, desc="validating", leave=False):
            reactants = reactants.to(self.device) if not isinstance(reactants, list) else reactants
            products = products.to(self.device) if not isinstance(products, list) else products
            labels = labels.to(self.device)
            logits = self.model(reactants, products)
            logits_list.append(logits.cpu())
            loss = self.loss_fn(logits, labels)
            for metric_fn in self.valid_metrics.values():
                metric_fn.update(logits.cpu(), labels.cpu().long())
            losses.append(loss.cpu().item())
        metrics = {"loss": np.mean(losses)}
        logits = torch.cat(logits_list, dim=0)
        torch.save(logits, self.run_dir / "logits.pt")

        for name, metric_fn in self.valid_metrics.items():
            value = metric_fn.compute()
            if isinstance(value, torch.Tensor):
                metrics[name] = value.item()
            elif isinstance(value, dict):
                metrics.update(value)
            metric_fn.reset()
        self.logger.log_metrics(metrics=metrics, prefix="valid")
        self.model.train()
        return metrics

    def make_checkpoint(self, checkpoint_name: str, metrics: Optional[Dict[str, float]] = None):
        checkpoint_dir = self.run_dir / "train" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint_dict, checkpoint_dir / f"{checkpoint_name}.pt")

    def train(self) -> Dict[str, float]:
        batch_no = 0
        losses = []
        for epoch in (epoch_pbar := tqdm(range(self.n_epochs), total=self.n_epochs)):
            epoch_pbar.set_description(f"epoch {epoch}")
            for reactants, products, labels in (
                pbar := tqdm(self.train_loader, desc=f"training", leave=False)
            ):
                self.optimizer.zero_grad()
                reactants = reactants.to(self.device)
                products = products.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(reactants, products)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)
                self.optimizer.step()

                losses.append(loss.item())
                for name, metric_fn in self.train_metrics.items():
                    metric_fn(logits.detach().cpu(), labels.cpu().long())

                if batch_no % self.log_train_every_n_batches == 0:
                    metrics = {"loss": np.mean(losses)}
                    for name, metric_fn in self.train_metrics.items():
                        metrics[name] = metric_fn.compute().item()
                        metric_fn.reset()
                    self.logger.log_metrics(metrics=metrics, prefix="train")
                    pbar.set_description(f"loss: {loss.item():.4f}")
                    losses = []
                batch_no += 1

            if epoch % self.valid_every_n_epochs == 0 or epoch == self.n_epochs - 1:
                valid_metrics = self.valid_step()
                if self.checkpoint_best:
                    if self.metric_direction == "min":
                        is_best = valid_metrics[self.best_metric] < self.best_valid_metrics.get(
                            self.best_metric, float("inf")
                        )
                    else:
                        is_best = valid_metrics[self.best_metric] > self.best_valid_metrics.get(
                            self.best_metric, float("-inf")
                        )
                    if is_best:
                        self.logger.log_metrics(metrics=valid_metrics, prefix="best_valid")
                        self.best_valid_metrics = valid_metrics | {"epoch": epoch}
                        self.make_checkpoint(
                            checkpoint_name="best_reaction", metrics=self.best_valid_metrics
                        )
                else:
                    self.best_valid_metrics = valid_metrics | {"epoch": epoch}

        self.make_checkpoint(checkpoint_name="last_reaction", metrics=self.best_valid_metrics)
        return self.best_valid_metrics

    def close(self):
        self.logger.close()
