"""For Simple-Model Trainer"""

from typing import List, Tuple
from logging import Logger
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from src import Trainer
from ohta_dataloader import OhtaDataloader


class OhtaTrainer(Trainer):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        dataloader: OhtaDataloader,
        valid_rate: float,
        model_save_path: str,
        *args,
        logger: Logger = None,
        save_per_epoch: bool = False,
        scaler: torch.cuda.amp.GradScaler = None,
        **kwargs
    ) -> None:
        super().__init__(
            net=net,
            criterion=None,
            optimizer=optimizer,
            dataloader=dataloader,
            valid_rate=valid_rate,
            model_save_path=model_save_path,
            logger=logger,
            save_per_epoch=save_per_epoch,
            *args,
            **kwargs
        )
        self.scaler = scaler

    def form_model_io(
        self, batch: List[List[List[torch.Tensor]]]
    ) -> Tuple[float, float]:
        source = batch[0]
        target = batch[1]

        batch_size = source[0][0].shape[0]

        with torch.cuda.amp.autocast():
            (_angl, _cent) = self.net(source)

            # loss_angl = torch.sqrt(torch.sum((_angl - target[0]) ** 2) / batch_size)
            # loss_cent = torch.sqrt(torch.sum((_cent - target[1]) ** 2) / batch_size)
            loss_angl = torch.sum(torch.abs(_angl - target[0])) / batch_size
            loss_cent = torch.sum(torch.abs(_cent - target[1])) / batch_size

            loss = loss_angl + loss_cent
            _loss = self.learn(loss)

        acc = torch.sum(abs(_angl - target[0]) + abs(_cent - target[1])) / batch_size
        _acc = float(acc)
        del acc

        return (_loss, _acc)

    def learn(self, loss: torch.Tensor):
        _loss = float(loss)
        if self.mode == "valid":
            del loss
            return _loss

        self.optimizer.zero_grad()

        if self.scaler is None:
            loss.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        del loss
        return _loss
