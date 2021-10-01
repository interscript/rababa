
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

from util import nakdimon_dataset # as dataset
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
            self.model, opt, self.global_step = self.config_manager.load_model()
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

        print('data:: ', path)
        dataset = DiacritizationDataset(self.config_manager, path)
        print('diti')
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

    def diacritize_data_iterator(self, data_iterator,
                                 criterion=None):
        raw_data, dia_data = [], []
        losses = []
        for data_batch in tqdm.tqdm(data_iterator):
            data_batch.to_device(self.device)
            raw_data.append(data_batch)
            preds, loss = self.predict_batch(data_batch, criterion)
            dia_data.append(preds)
            if not criterion is None:
                losses.append(loss)

        raw_data = nakdimon_dataset.Data.concatenate(raw_data)
        dia_data = nakdimon_dataset.Data.concatenate(dia_data)
        losses = [l[0] for l in losses], [l[1] for l in losses], [l[2] for l in losses]
        return raw_data, dia_data, losses


    def predict_batch(self, data_batch: nakdimon_dataset.Data,
                      criterion=None):

        def process_dim(dim):
            return niqqud.permute(0, 2, 1) # if self.device=='cpu' else \
        #        np.transpose(l_dia_data[i], (0, 2, 1))

        # Forward pass
        niqqud, dagesh, sin = self.model(data_batch.normalized)

        losses = None
        if not criterion is None:

            losses = [criterion(process_dim(niqqud), data_batch.niqqud.long()),
                      criterion(process_dim(dagesh), data_batch.dagesh.long()),
                      criterion(process_dim(sin), data_batch.sin.long())]

        return nakdimon_dataset.Data(data_batch.text, data_batch.normalized, \
                            torch.max(dagesh.permute(0, 2, 1), 1). \
                                    indices.detach().cpu().numpy(), \
                            torch.max(sin.permute(0, 2, 1), 1). \
                                    indices.detach().cpu().numpy(), \
                            torch.max(niqqud.permute(0, 2, 1), 1). \
                                            indices.detach().cpu().numpy()), \
                losses


    def diacritize_iterators(self, iterator):
        pass
