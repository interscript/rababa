from enum import Enum
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict

import ruamel.yaml
import torch

from models.baseline import BaseLineModel
from models.cbhg import CBHGModel


from options import AttentionType, LossType, OptimizerType
from util.text_encoders import (
    TextEncoder,
    # HebraicEncoder
)


class ConfigManager:
    """Config Manager"""

    def __init__(self, config_path: str, model_kind: str):
        available_models = [
            "baseline",
            "cbhg",
        ]
        if model_kind not in available_models:
            raise TypeError(f"model_kind must be in {available_models}")
        self.config_path = Path(config_path)
        self.model_kind = model_kind
        self.yaml = ruamel.yaml.YAML()
        self.config: Dict[str, Any] = self._load_config()
        self.set_device()
        self.session_name = ".".join(
            [self.config["session_name"], f"{model_kind}"]  # self.config["data_type"],
        )

        self.data_dir = Path(os.path.join(self.config["data_directory"]))

        self.base_dir = Path(
            os.path.join(self.config["log_directory"], self.session_name)
        )

        self.log_dir = Path(os.path.join(self.base_dir, "logs"))
        self.prediction_dir = Path(os.path.join(self.base_dir, "predictions"))
        self.plot_dir = Path(os.path.join(self.base_dir, "plots"))
        self.models_dir = Path(os.path.join(self.base_dir, "models"))

        self.text_encoder: TextEncoder = self.get_text_encoder()

        if self.model_kind in ["seq2seq", "tacotron_based"]:
            self.config["attention_type"] = AttentionType[self.config["attention_type"]]
        self.config["optimizer"] = OptimizerType[self.config["optimizer_type"]]

    def _load_config(self):
        with open(self.config_path, "rb") as model_yaml:
            _config = self.yaml.load(model_yaml)
        return _config

    def set_device(self):
        if self.config.get("device"):
            self.device = self.config["device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _get_git_hash():
        try:
            return (
                subprocess.check_output(["git", "describe", "--always"])
                .strip()
                .decode()
            )
        except Exception as e:
            print(f"WARNING: could not retrieve git hash. {e}")

    def _check_hash(self):
        try:
            git_hash = (
                subprocess.check_output(["git", "describe", "--always"])
                .strip()
                .decode()
            )
            if self.config["git_hash"] != git_hash:
                print(
                    f"""WARNING: git hash mismatch. Current: {git_hash}.
                    Config hash: {self.config['git_hash']}"""
                )
        except Exception as e:
            print(f"WARNING: could not check git hash. {e}")

    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * " "
        print(tab + "-", key_name, ":", values)

    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dictionary(dictionary[key], recursion_level)
            else:
                self._print_dict_values(
                    dictionary[key], key_name=key, level=recursion_level
                )

    def print_config(self):
        print("\nCONFIGURATION", self.session_name)
        self._print_dictionary(self.config)

    def update_config(self):
        self.config["git_hash"] = self._get_git_hash()

    def dump_config(self):
        self.update_config()
        _config = {}
        for key, val in self.config.items():
            if isinstance(val, Enum):
                _config[key] = val.name
            else:
                _config[key] = val
        with open(self.base_dir / "config.yml", "w") as model_yaml:
            self.yaml.dump(_config, model_yaml)

    def create_remove_dirs(
        self,
        clear_dir: bool = False,
        clear_logs: bool = False,
        clear_weights: bool = False,
        clear_all: bool = False,
    ):
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir.mkdir(exist_ok=True)
        self.prediction_dir.mkdir(exist_ok=True)

        if clear_dir:
            delete = input(f"Delete {self.log_dir} AND {self.models_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.log_dir, ignore_errors=True)
                shutil.rmtree(self.models_dir, ignore_errors=True)
        if clear_logs:
            delete = input(f"Delete {self.log_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.log_dir, ignore_errors=True)
        if clear_weights:
            delete = input(f"Delete {self.models_dir}? (y/[n])")
            if delete == "y":
                shutil.rmtree(self.models_dir, ignore_errors=True)
        self.log_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def get_last_model_path(self):
        """
        Given a checkpoint, get the last save model name
        Args:
        checkpoint (str): the path where models are saved
        """
        models = os.listdir(self.models_dir)
        # print(models)
        models = [model for model in models if model[-3:] == ".pt"]
        if len(models) == 0:
            return None
        _max = max(int(m.split(".")[0].split("-")[0]) for m in models)
        model_name = f"{_max}-snapshot.pt"
        last_model_path = os.path.join(self.models_dir, model_name)

        return last_model_path

    def load_model(self, model_path: str = None, load_optimizer: bool = False):
        """
        loading a model from path
        Args:
        model_path (str): the path to the model
        load_optimizer: load optimizer for training
        """

        model = self.get_model()

        with open(self.base_dir / f"{self.model_kind}_network.txt", "w") as file:
            file.write(str(model))

        model_path = self.config["model_path"] if model_path is None else model_path

        try:
            saved_model = (
                torch.load(model_path)
                if torch.cuda.is_available()
                else torch.load(model_path, map_location=torch.device("cpu"))
            )
            check = model.load_state_dict(saved_model["model_state_dict"])
            print("Load model state dict:: ", check)  # check...
            optimizer_stat_dict = (
                saved_model["optimizer_state_dict"] if load_optimizer else None
            )
            global_step = saved_model["global_step"] + 1

        except:
            print("model_path:: ", model_path)
            print("WARNING:: Model not found under model_state_dict,")
            print("starting with a fresh model.")
            optimizer_stat_dict = None
            global_step = 0

        model.to(self.device)

        return model, optimizer_stat_dict, global_step

    def get_model(self, ignore_hash=True):
        if not ignore_hash:
            self._check_hash()
        if self.model_kind == "cbhg":
            return self.get_cbhg()

        elif self.model_kind == "baseline":
            return self.get_baseline()

    def get_baseline(self):
        model = BaseLineModel(
            embedding_dim=self.config["embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            targ_vocab_size=self.config["len_target_symbols"],
            layers_units=self.config["layers_units"],
            use_batch_norm=self.config["use_batch_norm"],
        )

        return model

    def get_cbhg(self):
        model = CBHGModel(
            embedding_dim=self.config["embedding_dim"],
            inp_vocab_size=self.config["len_input_symbols"],
            targ_niqqud_size=self.config["len_niqqud_symbols"],
            targ_dagesh_size=self.config["len_dagesh_symbols"],
            targ_sin_size=self.config["len_sin_symbols"],
            use_prenet=self.config["use_prenet"],
            prenet_sizes=self.config["prenet_sizes"],
            cbhg_gru_units=self.config["cbhg_gru_units"],
            cbhg_filters=self.config["cbhg_filters"],
            cbhg_projections=self.config["cbhg_projections"],
            post_cbhg_layers_units=self.config["post_cbhg_layers_units"],
            post_cbhg_use_batch_norm=self.config["post_cbhg_use_batch_norm"],
        )

        return model

    def get_text_encoder(self):
        """Getting the class of TextEncoder from config"""
        """
        if self.config["text_cleaner"] not in [
            "basic_cleaners",
            "valid_hebraic_cleaners",
            None,
        ]:
            raise Exception(f"cleaner is not known {self.config['text_cleaner']}")
        """
        return TextEncoder(self.config)

    def get_loss_type(self):
        try:
            loss_type = LossType[self.config["loss_type"]]
        except:
            raise Exception(f"The loss type is not correct {self.config['loss_type']}")
        return loss_type
