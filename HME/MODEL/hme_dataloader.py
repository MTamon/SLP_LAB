"""For Simple-Model Dataloader"""

from __future__ import annotations
from typing import List, Dict
import torch

from .src import override
from .src import Dataloader
from .hme_dataset import HmeDataset


class HmeDataloader(Dataloader):
    def __init__(
        self,
        dataset: HmeDataset,
        batch_size: int,
        truncate_excess: bool,
        do_shuffle: bool,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset, batch_size, truncate_excess, do_shuffle, *args, **kwargs
        )

    @override(Dataloader)
    def collect_batch(
        self, batch_idx: List[int]
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        _batch = {
            "src": {"angl": [], "cent": [], "trgt": [], "othr": []},
            "target": {"angl": [], "cent": []},
        }
        for _idx in batch_idx:
            record = self.dataset[_idx]
            _batch["src"]["angl"].append(record[0][0])
            _batch["src"]["cent"].append(record[0][1])
            _batch["src"]["trgt"].append(record[0][2])
            _batch["src"]["othr"].append(record[0][3])

            _batch["target"]["angl"].append(record[0][0])
            _batch["target"]["cent"].append(record[0][1])

        return _batch

    @override(Dataloader)
    def copy(self) -> HmeDataloader:
        _copy = HmeDataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            truncate_excess=self.truncate_excess,
            do_shuffle=self.do_shuffle,
            *self.args,
            **self.kwargs,
        )
        return _copy
