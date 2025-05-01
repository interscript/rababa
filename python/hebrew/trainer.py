import os
from typing import Dict

import torch
from torch import nn
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from tqdm import trange
import numpy as np

from config_manager import ConfigManager
from dataset import load_iterators
from diacritizer import Diacritizer
from util.learning_rates import LearningRateDecay
from options import OptimizerType

from util.utils import (
    categorical_accuracy,
    count_parameters,
    # initialize_weights,
    # plot_alignment,
    repeater,
)

from util import nakdimon_dataset
from util import nakdimon_utils as utils
from util import nakdimon_hebrew_model as hebrew
from util import nakdimon_metrics

# Make wandb optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available in trainer.py, training will proceed without wandb logging")


class Trainer:
    def run(self):
        raise NotImplementedError


class GeneralTrainer(Trainer):
    def __init__(self, config_path: str, model_kind: str) -> None:

        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=self.config_path, model_kind=self.model_kind
        )
        self.config = self.config_manager.config
        self.losses = []
        self.lr, self.pad_idx = 0, 0
        self.device = self.config_manager.device

        self.config_manager.create_remove_dirs()
        self.text_encoder = self.config_manager.text_encoder
        self.summary_manager = SummaryWriter(log_dir=self.config_manager.log_dir)

        self.load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            weight_decay=self.config["weight_decay"],
        )

        self.criterion = nn.CrossEntropyLoss()  # ignore_index=self.pad_idx)

        self.load_diacritizer()
        self.diacritizer.set_model(self.model)
        # self.initialize_model()
        self.print_config()

    def print_config(self):
        self.config_manager.dump_config()
        self.config_manager.print_config()

        if self.global_step > 1:
            print(f"loaded form {self.global_step}")

        parameters_count = count_parameters(self.model)
        print(f"Model has {parameters_count} trainable parameters parameters")

    def load_diacritizer(self):
        if self.model_kind in ["cbhg", "baseline"]:
            load_model = False  # True
            self.diacritizer = Diacritizer(
                self.config_path, self.model_kind
            )  # , load_model)
        else:
            print("model not found")
            exit()

    def print_losses(self, step_results, tqdm):

        for pos, n_steps in enumerate(self.config["n_steps_avg_losses"]):
            if len(self.losses) > n_steps:
                d_losses = process_losses(step_results[-n_steps:])
                for k in d_losses.keys():

                    for i, k in enumerate(d_losses.keys()):
                        tqdm.display(
                            f"{n_steps}-steps average {k}_loss: {d_losses[k]}",
                            pos=pos + 3 + i,
                        )

    def process_losses(self, losses):
        n_losses = len(losses)
        d_loss = {}
        for k in ["N", "D", "S"]:
            loss = sum([l[k] for l in losses]) / n_losses
            d_loss[k + "_loss"] = loss
        return d_loss

    def get_benchmarks(self, test_data_iterator, dims=["N", "D", "S"]):  # tqdm

        d_scores = {}
        # Run the model on some test examples
        with torch.no_grad():

            raw_data, dia_data, losses = self.diacritizer.diacritize_data_iterator(
                test_data_iterator, self.criterion
            )

            l_raw_data = [raw_data.niqqud, raw_data.dagesh, raw_data.sin]
            l_dia_data = [dia_data.niqqud, dia_data.dagesh, dia_data.sin]

            for i, k in enumerate(dims):
                labels = l_raw_data[i].flatten()
                preds = l_dia_data[i].flatten()
                d_scores[k + "_accu"] = (labels == preds).sum().item() / preds.shape[0]
                d_scores[k + "_loss"] = float(sum(losses[i]) / len(losses[i]))

        return d_scores

    def evaluate_with_error_rates(self, iterator, tqdm):

        tqdm.set_description(f"Calculating DEC/CHA/WOR/VOC {self.global_step}: ")

        test_path = os.path.join(
            self.config_manager.data_dir,
            "test",
            self.config_manager.config["test_file_name"],
        )

        orig_path = os.path.join(self.config_manager.prediction_dir, f"original.txt")
        predicts_path = os.path.join(
            self.config_manager.prediction_dir, f"predicted.txt"
        )

        f = open(test_path, "r")
        all_orig = f.readlines()
        f.close()

        with open(orig_path, "w", encoding="utf8") as file:
            for sentence in all_orig:
                file.write(f"{sentence}")
        file.close()

        # diacritize and write to file
        self.diacritizer.diacritize_file(orig_path, predicts_path)

        # evaluate metrics
        results = nakdimon_metrics.all_metrics_for_files(orig_path, predicts_path)

        tqdm.reset()
        return results, None

    def run(self, config_wandb=None):

        scaler = torch.cuda.amp.GradScaler()
        train_iterator, _, validation_iterator = load_iterators(self.config_manager)
        n_steps_per_epoch = len(train_iterator)

        print("data loaded")
        print("----------------------------------------------------------")
        tqdm_eval = trange(0, len(validation_iterator), leave=True)
        tqdm_error_rates = trange(0, len(validation_iterator), leave=True)
        tqdm_eval.set_description("Eval")
        tqdm_error_rates.set_description("DEC/CHA/WOR/VOC : ")
        tqdm = trange(self.global_step, self.config["max_steps"] + 1, leave=True)
        print("--------------------------------------")

        for batch_inputs in repeater(train_iterator):

            tqdm.set_description(f"Global Step {self.global_step}")
            if self.config["use_decay"]:
                self.lr = self.adjust_learning_rate(
                    self.optimizer, global_step=self.global_step
                )

            self.optimizer.zero_grad()
            batch_inputs.to_device(self.device)
            step_results = self.train_batch(batch_inputs)

            if self.device == "cuda" and self.config["use_mixed_precision"]:
                with autocast():

                    for k in step_results.keys():
                        scaler.scale(step_results[k]).backward(retain_graph=True)

                    scaler.unscale_(self.optimizer)
                    if self.config.get("CLIP"):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["CLIP"]
                        )

                    scaler.step(self.optimizer)
                    scaler.update()
            else:

                for k in step_results.keys():
                    step_results[k].backward(retain_graph=True)

                if self.config.get("CLIP"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["CLIP"]
                    )
                self.optimizer.step()

            dico = {
                "N": float(step_results["N"]),
                "S": float(step_results["S"]),
                "D": float(step_results["D"]),
            }

            self.print_losses(step_results, tqdm)

            if self.global_step % self.config["model_save_frequency"] == 0:
                torch.save(
                    {
                        "global_step": self.global_step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(
                        self.config_manager.models_dir,
                        f"{self.global_step}-snapshot.pt",
                    ),
                )

            if self.global_step % n_steps_per_epoch == 0:

                self.diacritizer.set_model(self.model)
                d_scores = self.get_benchmarks(validation_iterator)

                scores, _ = self.evaluate_with_error_rates(
                    validation_iterator, tqdm_error_rates
                )

                if not config_wandb is None and WANDB_AVAILABLE:
                    wandb.log({**d_scores, **scores})
                    print("scores:: ", scores)

                else:

                    tqdm.display(
                        f"Evaluate {self.global_step}: N_accu, {d_scores['N_accu']}, N_loss: {d_scores['N_loss']}",
                        pos=8,
                    )
                    tqdm.display(
                        f"Evaluate {self.global_step}: D_accu, {d_scores['D_accu']}, D_loss: {d_scores['D_loss']}",
                        pos=9,
                    )
                    tqdm.display(
                        f"Evaluate {self.global_step}: S_accu, {d_scores['S_accu']}, S_loss: {d_scores['S_loss']}",
                        pos=10,
                    )
                    DEC, CHA, WOR, VOC = (
                        scores["dec"],
                        scores["cha"],
                        scores["wor"],
                        scores["voc"],
                    )

                    error_rates = f"DEC: {DEC}, CHA: {CHA}, WOR: {WOR}, VOC: {VOC}"
                    tqdm.display(f"metrics {self.global_step}: {error_rates}", pos=11)
                    # print('summray_texts:: ', summary_texts)

                    if scores:

                        """
                        self.summary_manager.add_scalar(
                            "error_rates/DEC", DEC, global_step=self.global_step)
                        self.summary_manager.add_scalar(
                            "error_rates/CHA", CHA, global_step=self.global_step)
                        self.summary_manager.add_scalar(
                            "error_rates/WOR", WOR, global_step=self.global_step)
                        self.summary_manager.add_scalar(
                            "error_rates/VOC", VOC, global_step=self.global_step)
                        """

            if self.global_step % self.config["train_plotting_frequency"] == 0:
                self.plot_attention(step_results)

            self.report(step_results, tqdm)

            self.global_step += 1
            if self.global_step > self.config["max_steps"]:
                print("Training Done.")
                return

            tqdm.update()

    def train_batch(self, raw_data: nakdimon_dataset.Data, labels=["N", "D", "S"]):

        # Forward pass
        targets = raw_data.niqqud, raw_data.dagesh, raw_data.sin

        outputs = self.model(raw_data.normalized)
        losses = []

        for i, k in enumerate(labels):
            # Evaluate loss
            loss = self.criterion(outputs[i].permute(0, 2, 1), targets[i].long())
            # Step with optimizer
            losses.append((labels[i], loss))

        return dict(losses)

    def predict(self, iterator):
        pass

    def load_model(self, model_path: str = None, load_optimizer: bool = True):

        (
            saved_model,
            optimizer_states_dict,
            global_step,
        ) = self.config_manager.load_model(model_path, load_optimizer)

        self.model = saved_model
        if not optimizer_states_dict is None:
            self.optimizer.load_state_dict(optimizer_states_dict)

        self.global_step = global_step if not global_step is None else 0

    def get_optimizer(self):
        if self.config["optimizer"] == OptimizerType.Adam:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == OptimizerType.SGD:
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9
            )
        else:
            raise ValueError("Optimizer option is not valid")

        return optimizer

    def get_learning_rate(self):
        return LearningRateDecay(
            lr=self.config["learning_rate"],
            warmup_steps=self.config.get("warmup_steps", 4000.0),
        )

    def adjust_learning_rate(self, optimizer, global_step):
        learning_rate = self.get_learning_rate()(global_step=global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        return learning_rate

    def plot_attention(self, results):
        pass

    def report(self, results, tqdm):
        pass


class CBHGTrainer(GeneralTrainer):
    pass
