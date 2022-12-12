"""For Simple-Model Trainer"""

from typing import List, Dict, Tuple
from logging import Logger
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.optim.optimizer import Optimizer

from train import Trainer
from hme_dataloader import HmeDataloader


class HmeTrainer(Trainer):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        dataloader: HmeDataloader,
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
        self, batch: Dict[str, Dict[str, List[torch.Tensor]]]
    ) -> Tuple[float, float]:
        src_angl, mask_a = self.padding(batch["src"]["angl"])
        src_cent, mask_c = self.padding(batch["src"]["cent"])
        src_trgt, _ = self.padding(batch["src"]["trgt"])
        src_othr, _ = self.padding(batch["src"]["othr"])
        trg_angl, _ = self.padding(batch["target"]["angl"])
        trg_cent, _ = self.padding(batch["target"]["cent"])

        with torch.cuda.amp.autocast():
            if self.mode == "train":
                model_input = [src_angl, src_cent, src_trgt, src_othr]
                (pred_angl, pred_cent), _ = self.net(model_input)
            else:
                pred_angl = torch.zeros_like(src_angl[:, 0:1, :])
                pred_cent = torch.zeros_like(src_cent[:, 0:1, :])
                h, c = None, None
                _angl = torch.unsqueeze(src_angl[:, 0, :], dim=1)
                _cent = torch.unsqueeze(src_cent[:, 0, :], dim=1)

                for _seq in range(src_angl.shape[1]):
                    _trgt = torch.unsqueeze(src_trgt[:, _seq, :], dim=1)
                    _othr = torch.unsqueeze(src_othr[:, _seq, :], dim=1)

                    model_input = [_angl, _cent, _trgt, _othr]
                    (_p_angl, _p_cent), (h, c) = self.net(model_input, h, c)

                    pred_angl = torch.cat((pred_angl, _p_angl), axis=1)
                    pred_cent = torch.cat((pred_cent, _p_cent), axis=1)

                    _angl, _cent = _p_angl, _p_cent

                pred_angl = pred_angl[:, 1:, :]
                pred_cent = pred_cent[:, 1:, :]

            pred_angl *= mask_a
            trg_angl *= mask_a
            pred_cent *= mask_c
            trg_cent *= mask_c

            div = torch.sum(mask_a) * 2

            loss_angl = torch.sum((pred_angl - trg_angl) ** 2) / div
            loss_cent = torch.sum((pred_cent - trg_cent) ** 2) / div

            loss = loss_angl + loss_cent
            self.learn(loss)

        acc = torch.sum(abs(pred_angl - trg_angl) + abs(pred_cent - trg_cent)) / div

        return (loss, acc)

    def learn(self, loss: torch.Tensor):
        if self.mode == "valid":
            return
        if self.scaler is None:
            loss.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def padding(self, data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = pack_sequence(data, enforce_sorted=False)
        batch, _ = pad_packed_sequence(packed, batch_first=True, padding_value=1e5)
        mask = batch != 1e5
        mask = mask.int()

        batch = batch.float().to(self.device)
        mask = mask.float().to(self.device)

        return (batch, mask)
