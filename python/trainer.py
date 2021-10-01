
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
        self.lr = 0
        self.pad_idx = 0
        self.device = self.config_manager.device

        self.config_manager.create_remove_dirs()
        self.text_encoder = self.config_manager.text_encoder
        self.summary_manager = SummaryWriter(log_dir=self.config_manager.log_dir)

        self.model = self.config_manager.get_model()

        self.optimizer = self.get_optimizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self.load_model() #model_path=self.config.get("train_resume_model_path"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.load_diacritizer()
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
            self.diacritizer = Diacritizer(self.config_path, self.model_kind)
        else:
            print('model not found')
            exit()

    """
    def initialize_model(self):
        if self.global_step > 1:
            return
        if self.model_kind == "transformer":
            print("Initializing using xavier_uniform_")
            self.model.apply(initialize_weights)
    """

    def print_losses(self, step_results, tqdm):
        #self.summary_manager.add_scalar(
        #    "loss/loss", step_results, global_step=self.global_step
        #)

        #tqdm.display(f"loss: {step_results['loss']}", pos=3)
        for pos, n_steps in enumerate(self.config["n_steps_avg_losses"]):
            if len(self.losses) > n_steps:
                d_losses = process_losses(step_results[-n_steps:])
                for k in d_losses.keys():
                    #self.summary_manager.add_scalar(
                    #   f"loss/loss-{n_steps}",
                    #    d_losses[k],
                    #    global_step=self.global_step,
                    #)
                    for i,k in enumerate(d_losses.keys()):
                        tqdm.display(
                            f"{n_steps}-steps average {k}_loss: {d_losses[k]}",
                            pos=pos + 3 + i,
                        )

    def process_losses(self, losses):
        n_losses = len(losses)
        d_loss = {}
        for k in ['N','D','S']:
            loss = sum([l[k] for l in losses])/n_losses
            d_loss[k+'_loss'] = loss
        return d_loss

    def get_benchmarks(self, test_data_iterator, dims = ['N','D','S']): # tqdm

        d_scores = {}
        # Run the model on some test examples
        with torch.no_grad():

            raw_data, dia_data, losses = \
                self.diacritizer.diacritize_data_iterator(test_data_iterator,
                                                          self.criterion)

            l_raw_data = [raw_data.niqqud, raw_data.dagesh, raw_data.sin]
            l_dia_data = [dia_data.niqqud, dia_data.dagesh, dia_data.sin]

            for i,k in enumerate(dims):
                labels = l_raw_data[i].flatten()
                preds = l_dia_data[i].flatten()
                d_scores[k+'_accu'] = \
                        (labels == preds).sum().item() / preds.shape[0]
                d_scores[k+'_loss'] = float(sum(losses[i]) / len(losses[i]))

        return d_scores

    """
    def evaluate(self, iterator, tqdm, use_target=True, d_targets=['D','S','N']):

        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        tqdm.set_description(f"Eval: {self.global_step}")
        with torch.no_grad():

            raw_data, pred_data = self.diacritizer. \
                                    diacritize_data_iterator(iterator)
            targets = [raw_data.dagesh, raw_data.sin, raw_data.niqqud]
            predicts = [pred_data.dagesh, pred_data.sin, pred_data.niqqud]
            for i.k in enumerate(d_targets):

            loss = self.criterion(predicts, targets.to(self.device))
            acc = categorical_accuracy(
                predicts, targets.to(self.device), self.pad_idx, self.device
            )
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            tqdm.update()

        tqdm.reset()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    """

    def evaluate_with_error_rates(self, iterator, tqdm):
        all_orig = []
        all_predicted = []
        results = {}
        self.diacritizer.set_model(self.model)
        evaluated_batches = 0
        tqdm.set_description(f"Calculating DEC/CHA/WOR/VOC {self.global_step}: ")

        """
        for batch in iterator:
            if evaluated_batches > int(self.config["error_rates_n_batches"]):
                break

            predicted = self.diacritizer.diacritize_batch(batch)
            all_predicted += predicted
            all_orig += batch["original"]
            tqdm.update()
        """
        #summary_texts = []
        orig_path = os.path.join(self.config_manager.prediction_dir,
                                 f"original.txt")
        predicted_path = os.path.join(self.config_manager.prediction_dir,
                                      f"predicted.txt")

        print('op:: ', orig_path)
        print('pp:: ', predicted_path)

        with open(orig_path, "w", encoding="utf8") as file:
            for sentence in all_orig:
                file.write(f"{sentence}\n")
        print('abc')

        self.diacritizer.diacritize_file(orig_path)

        with open(predicted_path, "w", encoding="utf8") as file:
            for sentence in all_predicted:
                file.write(f"{sentence}\n")

        results = nakdimon_metrics. \
                    all_metrics_for_files(test_file_path, tmp_path)
        print('results:: ', results)
        """
        for i in range(int(self.config["n_predicted_text_tensorboard"])):
            if i > len(all_predicted):
                break

            summary_texts.append(
                (f"eval-text/{i}", f"{ all_orig[i]} |->  {all_predicted[i]}")
            )
        """

        tqdm.reset()
        return results, summary_texts

    def run(self):

        scaler = torch.cuda.amp.GradScaler()
        train_iterator, _, validation_iterator = \
                        load_iterators(self.config_manager)
        print("data loaded")
        print("----------------------------------------------------------")

        tqdm_eval = trange(0, len(validation_iterator), leave=True)
        tqdm_error_rates = trange(0, len(validation_iterator), leave=True)
        tqdm_eval.set_description("Eval")
        tqdm_error_rates.set_description("DEC/CHA/WOR/VOC : ")
        tqdm = trange(self.global_step, self.config["max_steps"] + 1, leave=True)


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

            self.losses.append(step_results)


            self.print_losses(step_results, tqdm)

            self.summary_manager.add_scalar(
                "meta/learning_rate", self.lr, global_step=self.global_step
            )

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

            if self.global_step % self.config["evaluate_frequency"] == 0:
                #loss, acc = self.evaluate(validation_iterator, tqdm_eval)
                d_loss, d_acc = get_benchmarks(validation_iterator)

                self.summary_manager.add_scalar(
                    "evaluate/loss", loss, global_step=self.global_step
                )
                self.summary_manager.add_scalar(
                    "evaluate/acc", acc, global_step=self.global_step
                )
                tqdm.display(
                    f"Evaluate {self.global_step}: accuracy, {acc}, loss: {loss}", pos=8
                )
                self.model.train()

            if (
                self.global_step % self.config["evaluate_with_error_rates_frequency"]
                == 0
            ):
                scores, summery_texts = self.evaluate_with_error_rates(
                    validation_iterator, tqdm_error_rates
                )
                if error_rates:

                    DEC, CHA, WOR, VOC = \
                     scores["DEC"], scores["CHA"], scores["WOR"], scores["VOC"]

                    self.summary_manager.add_scalar(
                        "error_rates/DEC",
                        DEC,
                        global_step=self.global_step,
                    )
                    self.summary_manager.add_scalar(
                        "error_rates/CHA",
                        CHA,
                        global_step=self.global_step,
                    )
                    self.summary_manager.add_scalar(
                        "error_rates/WOR",
                        WOR,
                        global_step=self.global_step,
                    )
                    self.summary_manager.add_scalar(
                        "error_rates/VOC",
                        VOC,
                        global_step=self.global_step,
                    )

                    error_rates = f"DEC: {DEC}, CHA: {CHA}, WOR: {WOR}, VOC: {VOC}"
                    tqdm.display(f"metrics {self.global_step}: {error_rates}", pos=9)

                    for tag, text in summery_texts:
                        self.summary_manager.add_text(tag, text)

                self.model.train()

            if self.global_step % self.config["train_plotting_frequency"] == 0:
                self.plot_attention(step_results)

            self.report(step_results, tqdm)

            self.global_step += 1
            if self.global_step > self.config["max_steps"]:
                print("Training Done.")
                return

            tqdm.update()

    def train_batch(self, raw_data: nakdimon_dataset.Data,
                    labels = ['N', 'D', 'S']):
        # Forward pass
        #print(type(raw_data))
        #print('lllllllll:: ', raw_data.normalized.shape)
        #print('qqqqqqqqq:: ', type(raw_data.normalized))
        targets = raw_data.niqqud, raw_data.dagesh, raw_data.sin

        #print(11, raw_data.niqqud)
        #print(22, raw_data.dagesh)
        #print(33, raw_data.sin)

        outputs = self.model(raw_data.normalized)
        losses = []
        #self.optimizer.zero_grad()
        for i,k in enumerate(labels):
            # Evaluate loss
            loss = self.criterion(outputs[i].permute(0, 2, 1), \
                                  targets[i].long())
            # Backward pass
            #loss.backward(retain_graph=True)
            # Step with optimizer
            losses.append((labels[i], loss))
        #self.optimizer.step()
        return dict(losses)

    """
    def run_one_step(self, batch_inputs: Dict[str, torch.Tensor]):

        batch_inputs["src"] = batch_inputs["src"].to(self.device)
        batch_inputs["lengths"] = batch_inputs["lengths"].to("cpu")
        batch_inputs["target"] = batch_inputs["target"].to(self.device)

        outputs = self.model(
            src=batch_inputs["src"],
            target=batch_inputs["target"],
            lengths=batch_inputs["lengths"],
        )

        predictions = outputs["diacritics"].contiguous()
        targets = batch_inputs["target"].contiguous()

        predictions = predictions.view(-1, predictions.shape[-1])
        targets = targets.view(-1)

        loss = self.criterion(predictions.to(self.device),
                              targets.to(self.device))
        outputs.update({"loss": loss})
        return outputs
    """

    def predict(self, iterator):
        pass

    def load_model(self,
                   model_path: str = None,
                   load_optimizer: bool = True):

        saved_model, optimizer_states_dict, global_step = \
            self.config_manager.load_model(model_path, load_optimizer)

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
