from typing import Dict
import torch
from config_manager import ConfigManager


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
        batch_data = {'original': text, 
                      'src': torch.Tensor([seq]).long(),
                      'lengths': torch.Tensor([len(seq)]).long()}
        
        return self.diacritize_batch(batch_data)[0]
    
        def get_data_from_file(self, path):
            """get data from relative path"""
            loader_params = {"batch_size": self.config_manager.config["batch_size"],
                             "shuffle": False,
                             "num_workers": 2}
            # data processed or not, specs in config file
            if self.config_manager.config["is_data_preprocessed"]:
                data = pd.read_csv(path,
                                   encoding="utf-8",
                                   sep=self.config_manager.config["data_separator"],
                                   nrows=self.config_manager.config["n_validation_examples"],
                                   header=None)

                # data = data[data[0] <= config_manager.config["max_len"]]
                dataset = DiacritizationDataset(self.config_manager, data.index, data)
            else:
                with open(path, encoding="utf8") as file:
                    data = file.readlines()
                data = [text for text in data if len(text) <= self.config_manager.config["max_len"]]
                dataset = DiacritizationDataset(
                    self.config_manager, [idx for idx in range(len(data))], data
                    )

            data_iterator = DataLoader(dataset, collate_fn=collate_fn, **loader_params)
            # print(f"Length of data iterator = {len(valid_iterator)}")
            return data_iterator 
    
    def diacritize_file(self, path: str):
        """download data from relative path and diacritize it batch by batch"""
        data_iterator = self.get_data_from_file(path)
        diacritized_data = []
        for batch_inputs in data_iterator:
            
            batch_inputs["src"] = batch_inputs["src"].to(self.device)
            batch_inputs["lengths"] = batch_inputs["lengths"].to('cpu')
            batch_inputs["target"] = batch_inputs["target"].to(self.device)
            
            for d in self.diacritize_batch(batch_inputs):
                diacritized_data.append(d) 

        return diacritized_data

    def diacritize_batch(self, batch):
        raise NotImplementedError()

    def diacritize_iterators(self, iterator):
        pass


class CBHGDiacritizer(Diacritizer):
    
    def diacritize_batch(self, batch):
        #print('batch: ',batch)
        self.model.eval()
        inputs = batch["src"]
        lengths = batch["lengths"]
        outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)

        return sentences


class Seq2SeqDiacritizer(Diacritizer):
    
    def diacritize_batch(self, batch):
        self.model.eval()
        inputs = batch["src"]
        lengths = batch["lengths"]
        outputs = self.model(inputs.to(self.device), lengths.to("cpu"))
        diacritics = outputs["diacritics"]
        predictions = torch.max(diacritics, 2).indices
        sentences = []

        for src, prediction in zip(inputs, predictions):
            sentence = self.text_encoder.combine_text_and_haraqat(
                list(src.detach().cpu().numpy()),
                list(prediction.detach().cpu().numpy()),
            )
            sentences.append(sentence)

        return sentences
