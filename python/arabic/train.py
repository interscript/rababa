
import argparse
import random
import multiprocessing

import numpy as np
import torch

from trainer import (
    CBHGTrainer
)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_kind", dest="model_kind", type=str, required=True)
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument(
        "--reset_dir",
        dest="clear_dir",
        action="store_true",
        help="deletes everything under this config's folder.",
    )
    return parser


def main():
    parser = train_parser()
    args = parser.parse_args()

    if args.model_kind in ['baseline',"cbhg"]:
        trainer = CBHGTrainer(args.config, args.model_kind)
    else:
        raise ValueError("The model kind is not supported")

    trainer.run()


if __name__ == "__main__":
    # Fix for Python 3.9+ multiprocessing issues
    multiprocessing.freeze_support()
    main()
