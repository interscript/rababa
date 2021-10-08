"""
Loading the diacritization dataset
"""

import os
import numpy as np
import pandas as pd
import torch
import random
import warnings

from torch.utils.data import DataLoader, Dataset

from config_manager import ConfigManager

from util import nakdimon_dataset
from util import nakdimon_utils as utils
from util import nakdimon_hebrew_model as hebrew


class DiacritizationDataset(Dataset):
    """
    The datasets to preprocess for diacritization
    """

    def __init__(self, config_manager: ConfigManager, data_file_path):
        "Initialization"

        self.config = config_manager.config
        self.device = config_manager.device
        self.data_file_path = data_file_path

        with utils.smart_open(self.data_file_path, 'r', encoding='utf-8') as f:
            text = f.read() # hebrew.remove_niqqud(f.read())

        self.data = nakdimon_dataset.Data.from_text(hebrew.iterate_dotted_text(text), \
                                                    self.config['max_len'])

        # self.data, _ = nakdimon_dataset.get_data([self.data_file_path],
        #                                          self.config['max_len'])
        print('self.device:: ', self.device)
        self.data.to_device(self.device)


    def __len__(self):
        "Denotes the total number of samples"
        return self.data.size()

    def __getitem__(self, index):
        "Generates one sample of data"
        return self.data.get_id(index)


def collate_fn(data):
    """
    Padding the input and output sequences
    """
    return nakdimon_dataset.Data.concatenate(data)


def load_training_data(config_manager: ConfigManager, loader_parameters):
    """
    Loading the training data using pandas
    """
    if not config_manager.config["load_training_data"]:
        return []

    path = os.path.join(config_manager.data_dir, 'train', \
                        config_manager.config['train_file_name'])

    training_set = DiacritizationDataset(config_manager, path)

    train_iterator = DataLoader(
        training_set.data, collate_fn=collate_fn, **loader_parameters
    )

    print(f"Length of training iterator = {len(train_iterator)}")
    return train_iterator


def load_test_data(config_manager: ConfigManager, loader_parameters):
    """
    Loading the test data using pandas
    """
    if not config_manager.config["load_test_data"]:
        return []
    # test_file_name = config_manager.config.get("test_file_name", "test.csv")
    path = os.path.join(config_manager.data_dir, 'test',
                        config_manager.config['test_file_name'])

    test_dataset = DiacritizationDataset(config_manager, path)

    test_iterator = DataLoader(test_dataset.data, collate_fn=collate_fn,
                               **loader_parameters)

    print(f"Length of test iterator = {len(test_iterator)}")
    return test_iterator


def load_validation_data(config_manager: ConfigManager, loader_parameters):
    """
    Loading the validation data using pandas
    """

    if not config_manager.config["load_validation_data"]:
        return []
    # valid_file_name = config_manager.config.get("valid_file_name", "eval.txt")
    path = os.path.join(config_manager.data_dir, 'eval', \
                        config_manager.config['eval_file_name'])

    valid_dataset = DiacritizationDataset(config_manager, path)

    valid_iterator = DataLoader(
        valid_dataset.data, collate_fn=collate_fn, **loader_parameters
    )

    print(f"Length of valid iterator = {len(valid_iterator)}")
    return valid_iterator


def load_iterators(config_manager: ConfigManager):
    """
    Load the data iterators
    Args:
    """
    params = {
        "batch_size": config_manager.config["batch_size"],
        "shuffle": True,
        "num_workers": 0,
    }

    train_iterator = load_training_data(config_manager, loader_parameters=params)
    valid_iterator = load_validation_data(config_manager, loader_parameters=params)
    test_iterator = load_test_data(config_manager, loader_parameters=params)

    return train_iterator, test_iterator, valid_iterator
