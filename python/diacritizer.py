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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_model:
            self.model, self.global_step = self.config_manager.load_model()
            self.model = self.model.to(self.device)

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def diacritize_text(self, text: str):
        # convert string into indices
        text = text.strip()
        batch_data = self.text_encoder.str_to_data(text)
        return self.diacritize_batch(batch_data)[0]

    def get_data_from_file(self, path):
        """get data from relative path"""
        loader_params = {"batch_size": self.config_manager.config["batch_size"],
                         "shuffle": False,
                         "num_workers": 2}

        data_tmp = pd.read_csv(path,
                           encoding="utf-8",
                           sep=self.config_manager.config["data_separator"],
                           header=None)

        data = []
        max_len = self.config_manager.config["max_len"]
        for txt in [d[0] for d in data_tmp.values.tolist()]:
            if len(txt) > max_len:
                txt = txt[:max_len]
                warnings.warn('Warning: text length cut for sentence: \n'+txt)
            data.append(txt)

        list_ids = [idx for idx in range(len(data))]
        dataset = DiacritizationDataset(self.config_manager,
                                        list_ids,
                                        data)

        data_iterator = DataLoader(dataset,
                                   collate_fn=collate_fn,
                                   # **loader_params,
                                   shuffle=False)

        # print(f"Length of data iterator = {len(data_iterator)}")
        return data_iterator

    def diacritize_file(self, path: str):
        """download data from relative path and diacritize it batch by batch"""
        data_iterator = self.get_data_from_file(path)
        diacritized_data = []
        for batch_inputs in tqdm.tqdm(data_iterator):

            #batch_inputs["original"] = batch_inputs["original"].to(self.device)
            batch_inputs["src"] = batch_inputs["src"].to(self.device)
            batch_inputs["lengths"] = batch_inputs["lengths"].to('cpu')
            batch_inputs["target"] = batch_inputs["target"].to(self.device)

            for d in self.diacritize_batch(batch_inputs):
                diacritized_data.append(d)

        return diacritized_data

    def diacritize_batch(self, batch):
        # print('batch: ',batch)
        # self.model.eval()

        batch_size = config['batch_size']
        normalized = torch.tensor(data.normalized).to('cuda').long()
        n_data = normalized.shape[0]

        # niqqud, sin, dagesh = predict_batch(model, hebrew_idces)
        niqqud_sin_dagesh = [predict_batch(model, normalized[i:i+batch_size])
                             for i in range(0, n_data, batch_size)]

        if n_data % batch_size != 0:
            idx = int(n_data/batch_size) * batch_size
            niqqud_sin_dagesh += [predict_batch(model, normalized[idx:])]

        niqqud = np.concatenate([x[0] for x in niqqud_sin_dagesh])
        dagesh = np.concatenate([x[1] for x in niqqud_sin_dagesh])
        sin = np.concatenate([x[2] for x in niqqud_sin_dagesh])


        return normalized, niqqud, dagesh, sin

    def predict_batch(self, model, hebrew_normalized_idces):
        niqqud, dagesh, sin = model(hebrew_normalized_idces)
        return torch.max(niqqud.permute(0, 2, 1), 1).indices.detach().cpu().numpy(), \
                torch.max(dagesh.permute(0, 2, 1), 1).indices.detach().cpu().numpy(), \
                 torch.max(sin.permute(0, 2, 1), 1).indices.detach().cpu().numpy()

    def diacritize_iterators(self, iterator):
        pass
