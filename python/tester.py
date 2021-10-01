from config_manager import ConfigManager
import os
import torch
from typing import Dict

from torch import nn
from tqdm import tqdm
from tqdm import trange

from dataset import load_iterators
from trainer import GeneralTrainer

from util import nakdimon_dataset
from util import nakdimon_utils as utils
from util import nakdimon_hebrew_model as hebrew


class DiacritizationTester(GeneralTrainer):
    def __init__(self, config_path: str, model_kind: str) -> None:

        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=config_path, model_kind=model_kind
        )
        self.config = self.config_manager.config
        self.pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.device = self.config_manager.device

        self.text_encoder = self.config_manager.text_encoder

        self.model = self.config_manager.get_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.load_model(model_path=self.config["test_model_path"],
                        load_optimizer=False)
        self.model, opt, self.global_step =  \
            self.config_manager.load_model(model_path=self.config["test_model_path"],
                                           load_optimizer=False)
        self.model = self.model.to(self.device)
        self.load_diacritizer()
        self.diacritizer.set_model(self.model)

        # self.initialize_model()
        self.print_config()

    def run(self):

        self.config_manager.config["load_training_data"] = False
        self.config_manager.config["load_validation_data"] = False
        self.config_manager.config["load_test_data"] = True

        _, test_iterator, _ = load_iterators(self.config_manager)
        tqdm_eval = trange(0, len(test_iterator), leave=True)
        tqdm_error_rates = trange(0, len(test_iterator), leave=True)

        loss, acc = self.evaluate(test_iterator, tqdm_eval)
        error_rates, _ = self.evaluate_with_error_rates(test_iterator, tqdm_error_rates)

        tqdm_eval.close()
        tqdm_error_rates.close()

        WER = error_rates["WER"]
        DER = error_rates["DER"]
        DER1 = error_rates["DER*"]
        WER1 = error_rates["WER*"]

        error_rates = f"DER: {DER}, WER: {WER}, DER*: {DER1}, WER*: {WER1}"

        print(f"global step : {self.global_step}")
        print(f"Evaluate {self.global_step}: accuracy, {acc}, loss: {loss}")
        print(f"WER/DER {self.global_step}: {error_rates}")
