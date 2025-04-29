import argparse
from diacritizer import Diacritizer
from itertools import repeat
import random
import multiprocessing

import numpy as np
import torch


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def diacritization_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_kind", dest="model_kind", type=str, required=True)
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--text", dest="text", type=str, required=False)
    parser.add_argument("--text_file", dest="text_file", type=str, required=False)
    parser.add_argument(
        "--diacritized_text_file",
        dest="diacritized_text_file",
        type=str,
        required=False,
    )
    return parser


def main():
    parser = diacritization_parser()
    args = parser.parse_args()

    if args.text is None and args.text_file is None:
        raise ValueError("text or text_file/diacritized_text_file params required!")

    if args.model_kind == "cbhg":
        diacritizer = Diacritizer(args.config, args.model_kind, "log_dir")
    elif args.model_kind == "baseline":
        diacritizer = Diacritizer(args.config, args.model_kind, "log_dir")
    else:
        raise ValueError("The model kind is not supported")

    if args.text_file is None:
        txt = diacritizer.diacritize_text(args.text)
        print(txt)
    else:
        diacritizer.diacritize_file(args.text_file, args.diacritized_text_file)
        print("done!!! written in: ", args.diacritized_text_file)


if __name__ == "__main__":
    # Fix for Python 3.9+ multiprocessing issues
    multiprocessing.freeze_support()
    main()
