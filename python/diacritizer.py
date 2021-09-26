
from typing import Dict
import torch
import warnings
import tqdm
import pandas as pd
import numpy as np

from config_manager import ConfigManager
from dataset import (DiacritizationDataset,
                     collate_fn)
from torch.utils.data import (DataLoader,
                              Dataset)

from util_nakdimon import nakdimon_dataset as dataset
# import util.reconcile_original_plus_diacritized as reconcile


class Diacritizer:
    def __init__(
        self, config_path: str, model_kind: str, load_model: bool = False
    ) -> None:
        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=config_path, model_kind=model_kind
        )
        self.config = self.config_manager.config
        self.text_encoder = self.config_manager.text_encoder
        self.device = self.config_manager.device

        if load_model:
            self.model, self.global_step = self.config_manager.load_model()
            self.model = self.model.to(self.device)

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def diacritize_text(self, text: str):
        # convert string into indices
        text = text.strip()
        raw_data = dataset.Data.from_text(text)
        dia_data = self.predict_batch(data_batch)

        dia_total = dataset.merge_unconditional( \
                                raw_data.text, raw_data.normalized, \
                                dia_data.niqqud, dia_data.dagesh, dia_data.sin)

        text = ' '.join(dia_total).replace('\ufeff', ''). \
                        replace('  ', ' ').replace(hebrew.RAFE, '')
        return text

    def get_data_from_file(self, path):
        """get data from relative path"""
        loader_params = {"batch_size": self.config_manager.config["batch_size"],
                         "shuffle": False,
                         "num_workers": 2}

        dataset = DiacritizationDataset(path)

        data_iterator = DataLoader(dataset.data,
                                   collate_fn=collate_fn,
                                   **loader_params,
                                   shuffle=False)

        print(f"Length of data iterator = {len(data_iterator)}")
        return data_iterator

    def diacritize_file(self, path: str):
        """
            download data from relative path and diacritize it batch by batch
        """
        data_iterator = self.get_data_from_file(path)

        raw_data, dia_data = diacritize_data_iterator(self, data_iterator)

        dia_total = dataset.merge_unconditional( \
                                raw_data.text, raw_data.normalized, \
                                dia_data.niqqud, dia_data.dagesh, dia_data.sin)

        text = ' '.join(dia_total).replace('\ufeff', ''). \
                        replace('  ', ' ').replace(hebrew.RAFE, '')
        return text

    def diacritize_data_iterator(self, data_iterator): #model):
        aw_data, diacritized_data = [], []
        for data_batch in tqdm.tqdm(data_iterator):
            raw_data.append(data_batch)
            dia_data.append(self.predict_batch(data_batch))

        raw_data = Data.concatenate(raw_data)
        dia_data = Data.concatenate(dia_data)
        return raw_data, dia_data


    def predict_batch(self, data_batch: dataset.Data):
        # Forward pass
        niqqud, dagesh, sin = self.model(data_batch.normalized)

        return dataset.Data(data_batch.text, data_batch.normalized, \
                            torch.max(dagesh.permute(0, 2, 1), 1). \
                                    indices.detach().cpu().numpy(), \
                            torch.max(sin.permute(0, 2, 1), 1). \
                                    indices.detach().cpu().numpy(), \
                            torch.max(niqqud.permute(0, 2, 1), 1). \
                                            indices.detach().cpu().numpy())

    def diacritize_iterators(self, iterator):
        pass
