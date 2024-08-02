import pickle
from pathlib import Path
from typing import Any, Dict, List

import gin
import pandas as pd
import torch
from torch import Tensor
from torchmetrics import AUROC, Metric


@gin.configurable()
class SubsetMetrics(Metric):
    def __init__(self, metrics: Dict[str, Metric], subset_columns: List[str], split_path: str):
        super().__init__()
        self.preds = []
        self.targets = []
        self.metrics = metrics
        self.subset_columns = subset_columns
        self.split_path = split_path
        self.df = pd.read_csv(split_path)
        self.subset_columns_to_mask = {
            column: torch.tensor(self.df[column], dtype=torch.bool) for column in subset_columns
        }

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Any:
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        metric_dict = {}
        for column, mask in self.subset_columns_to_mask.items():
            for metric_name, metric_fn in self.metrics.items():
                metric_dict[f"{column}/{metric_name}"] = metric_fn(
                    preds[mask], targets[mask]
                ).item()
                metric_fn.reset()
        return metric_dict

    def reset(self) -> None:
        # dump preds and targets
        with open("dump_rfm.pkl", "wb") as f:
            pickle.dump((self.preds, self.targets), f)
        self.preds = []
        self.targets = []


@gin.configurable()
class MeanRank:
    def __call__(self, preds: Tensor, target: Tensor) -> Tensor:
        feasible_number = target.sum()
        chunk_size = len(target) // feasible_number
        assert feasible_number * chunk_size == len(target)

        preds = preds.reshape(-1, chunk_size)
        ranks = torch.argsort(preds, dim=1, descending=True)
        ranks = torch.argmin(ranks, dim=1) + 1
        return ranks.float().mean()

    def reset(self) -> None:
        pass


@gin.configurable()
class RankAccuracy:
    def __call__(self, preds: Tensor, target: Tensor) -> Tensor:
        feasible_number = target.sum()
        assert feasible_number * 2 == len(target)
        feasible_scores = preds[target == 1]
        infeasible_scores = preds[target == 0]
        print(len(feasible_scores), len(infeasible_scores))
        return (feasible_scores > infeasible_scores).float().mean() + (
            feasible_scores == infeasible_scores
        ).float().mean() * 0.5

    def reset(self) -> None:
        pass
