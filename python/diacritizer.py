from typing import Dict
import torch
import tqdm
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

        with open(path, encoding="utf8") as file:
            data = file.readlines()
        data = [text for text in data
                if len(text) <= self.config_manager.config["max_len"]]

        dataset = DiacritizationDataset(self.config_manager,
                                        [idx for idx in range(len(data))],
                                        data)

        data_iterator = DataLoader(dataset,
                                   collate_fn=collate_fn,
                                   **loader_params)

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
        #print('batch: ',batch)
        self.model.eval()
        originals = batch['original']
        inputs = batch["src"]
        lengths = batch["lengths"]
        outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction, original in zip(inputs, predictions, originals):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            # Diacritized strings, sentence have to be "reconciled"
            # with original strings, because the non arabic strings are removed
            # before being processed in nnet
            if self.config['reconcile']:
                sentence = reconcile.reconcile_strings(original, sentence)
            sentences.append(sentence)

        return sentences

    def diacritize_iterators(self, iterator):
        pass

""" not needed
class CBHGDiacritizer(Diacritizer):
class Seq2SeqDiacritizer(Diacritizer):
"""
