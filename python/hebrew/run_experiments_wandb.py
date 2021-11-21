
import argparse
import random

import numpy as np
import torch

# import ruamel.yaml
import ruamel.yaml as yaml

import wandb

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


parser = train_parser()
args = parser.parse_args()
    
# Define Experiments using Wandb
sweep_config = {
    # search method
    'method': 'random', #grid, random
    # metric and objective
    'metric': {
      'name': 'dec',
      'goal': 'maximize' #'minimize'   
    },
    # define search parameters
    'parameters': {
        'max_steps': {
            'values': [1000]
        },
        'batch_size': {
            'values': [32] #[128, 64, 32]
        },
        'cbhg_filters': {
            'values': [16]
        },
        'cbhg_gru_units': {
            'values': [256]
        },
        'cbhg_projections': {
            'values': [[128, 256]] #, [256, 512]]
        },

        'post_cbhg_layers_units': {
            'values': [[256, 256]]
        },
        
        'optimizer': {
            'values': ['Adam', 'SGD']
        },
        'use_prenet': {
            'values': ['false']
        },
        'prenet_sizes': {
            'values': [[512, 256]]
        }     
    }
}


# train code, with the search preprocessing logic    
def train():

    with open('config/train.yml', "rb") as model_yaml:
        config = yaml.load(model_yaml)
    
    # load default config
    config_defaults = config 
    wandb.init(config=config_defaults) # , magic=True)
    config_wandb = wandb.config
    
    # overwrite initial config
    config = { **config, 
               **config_wandb }
    
    tmp_config_path = 'config/sweep_tmp.yml'
    with open(tmp_config_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    if args.model_kind in ['baseline',"cbhg"]:
        trainer = CBHGTrainer(tmp_config_path, args.model_kind)
    else:
        raise ValueError("The model kind is not supported")

    trainer.run(config_wandb)
    

    
##################################
# MAIN                           #
##################################

# Run name
run_name = "hyperparams search"

# Init wandb and search
wandb.login()
sweep_id = wandb.sweep(sweep_config, project=run_name)

# Run search
wandb.agent(sweep_id, train)
