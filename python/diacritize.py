import argparse
#from diacritizer import TransformerDiacritizer
from diacritizer import (Diacritizer, CBHGDiacritizer)
from itertools import repeat
import random

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
    parser.add_argument("--text", dest="text", type=str, required=True)
    return parser


parser = diacritization_parser()
args = parser.parse_args()


if args.model_kind == "cbhg":
    diacritizer = CBHGDiacritizer(args.config, args.model_kind, 'log_dir')

elif args.model_kind == "baseline":
    diacritizer = Seq2SeqDiacritizer(args.config, args.model_kind, 'log_dir')
else:
    raise ValueError("The model kind is not supported")

diacritizer.diacritize_text(args.text)
