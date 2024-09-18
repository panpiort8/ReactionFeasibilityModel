import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

dir_path = Path(__file__).parent.absolute()
sys.path.append(str(dir_path))

import gin
from gin_config import get_time_stamp


@gin.configurable()
def train(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    batch_size: int,
    num_workers: int,
):
    persistent_workers = True if num_workers > 0 else False
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=valid_dataset.collate,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)

    args = parser.parse_args()
    config = args.cfg

    config_name = Path(config).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    gin.parse_config_files_and_bindings([config], bindings=[f'run_name="{run_name}"'])
    run_dir = gin.get_bindings("run_dir/macro")["value"]
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    train()
