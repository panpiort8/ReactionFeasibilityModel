import argparse
import sys
from pathlib import Path

dir_path = Path(__file__).parent.absolute()
sys.path.append(str(dir_path))

import gin
from gin_config import get_time_stamp
from torch_geometric import seed_everything
from trainer import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    config = args.cfg

    seed_everything(seed)
    config_name = Path(config).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    gin.parse_config_files_and_bindings([config], bindings=[f'run_name="{run_name}"'])
    trainer = ReactionTrainer()
    trainer.logger.log_code("external/reaction_feasibility")
    trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    trainer.logger.log_to_file(gin.config_str(), "config")
    trainer.train()
    trainer.close()
