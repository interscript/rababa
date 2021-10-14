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
import util.reconcile_original_plus_diacritized as reconcile


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

        self.start_symbol_id = self.text_encoder.start_symbol_id

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def diacritize_text(self, text: str):
        # convert string into indices
        text = text.strip()
        seq = self.text_encoder.input_to_sequence(text)
        # transform indices into "batch data"
        batch_data = {'original': [text],
                      'src': torch.Tensor([seq]).long(),
                      'lengths': torch.Tensor([len(seq)]).long()}

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
        self.model.eval()
        originals = batch['original']
        inputs = batch["src"]
        lengths = batch["lengths"]
        d_outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        # diacritics = outputs["diacritics"]
        # predictions = torch.max(diacritics, 2).indices
        pred_haraqat = d_outputs["haraqat"]
        preds_haraqat = torch.max(pred_haraqat, 2).indices
        pred_fatha = d_outputs["fatha"]
        preds_fatha = torch.max(pred_fatha, 2).indices
        pred_shadda = d_outputs["shaddah"]
        preds_shaddah = torch.max(pred_shaddah, 2).indices

        #d_predictions = {'haraqat': list(preds_haraqat.detach().cpu().numpy()),
        #                 'fatha': list(preds_fatha.detach().cpu().numpy()),
        #                 'shaddah': list(preds_shaddah.detach().cpu().numpy())}
        # for k in ['shaddah', 'shaddah', 'fatha']
        l_d_predictions =  [{'haraqat': h, 'faddah': f, 'shaddah': s} for h,f,s in
                zip(list(preds_haraqat.detach().cpu().numpy()),
                    list(preds_fatha.detach().cpu().numpy()),
                    list(preds_shaddah.detach().cpu().numpy()))]

        sentences = []
        for src, prediction, original in zip(inputs, l_d_predictions, originals):
            sentence = self.text_encoder.combine_text_and_haraqat(
                                                list(src.detach().cpu().numpy()),
                                                prediction)
            # Diacritized strings, sentence have to be "reconciled"
            # with original strings, because the non arabic strings are removed
            # before being processed in nnet
            if self.config['reconcile']:
                sentence = reconcile.reconcile_strings(original, sentence)
            sentences.append(sentence)

        return sentences

    def diacritize_iterators(self, iterator):
        pass
