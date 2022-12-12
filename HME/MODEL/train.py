"""This program is for training Simple-Model"""

from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Dict, Any, Tuple, Iterable
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from dataloader import Dataloader


class Trainer(metaclass=ABCMeta):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        dataloader: Dataloader,
        valid_rate: float,
        model_save_path: str,
        *args,
        logger: Logger = None,
        save_per_epoch: bool = False,
        **kwargs,
    ) -> None:
        self.args = args
        self.kwargs = kwargs

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

        self.dataloader = dataloader
        self.valid_rate = valid_rate
        self.model_save_path = model_save_path
        self.logger = logger
        self.save_per_epoch = save_per_epoch

        self.train_valid_loader = self.dataloader.train_valid(valid_rate)
        self.mode = "train"
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        net.to(device=self.device)
        self._logging(f"Device :{self.device}")
        self.reset()

    def __call__(self) -> Dict[str, Dict[str, List[float]]]:
        score = {"train": {"loss": [], "acc": []}, "valid": {"loss": [], "acc": []}}

        for _mode in ["train", "valid"]:
            self.set_mode(_mode)

            for loss, acc in self:
                score[_mode]["loss"].append(loss)
                score[_mode]["acc"].append(acc)

        return score

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[float, float]:
        try:
            batch = next(self.train_valid_loader[self.mode])
        except StopIteration:
            if self.mode == "train" and self.save_per_epoch:
                self.save_model()
            raise

        return self.form_model_io(batch)

    def __len__(self):
        return len(self.train_valid_loader[self.mode])

    def _logging(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def set_mode(self, mode=None):
        if not mode in ["train", "valid"]:
            raise ValueError(f"'mode' must be 'train' or 'valid', but {mode}")
        if self.train_valid_loader[self.mode].current_idx != 0:
            self._logging(f"! Remain batches in the '{self.mode}' Dataloader.")
            self.train_valid_loader[self.mode].reset()

        self.mode = mode

        if mode == "train":
            self.net.train()
        else:
            self.net.eval()

    def save_model(self, path: str = None):
        if path is None:
            torch.save(self.net.state_dict(), self.model_save_path)
        else:
            torch.save(self.net.state_dict(), path)

    @staticmethod
    def load_model(model: nn.Module, path: str, device: str = None) -> nn.Module:
        net_dic = torch.load(path, map_location=device)
        model.load_state_dict(net_dic)

        return model

    def set_model(self, model: nn.Module):
        self.net = model.to(device=self.device)

    def reset(self):
        torch.manual_seed(0)

    @abstractmethod
    def form_model_io(self, batch: Iterable[Any]) -> Tuple[float, float]:
        """Format the model input & result and return 'loss' and 'accuracy'.

        Args:
            batch (List[Any]): Input to model (self.net)

        Returns:
            Tuple[float, float]: loss (float) & accuracy (float).
        """
