"""For Simple-Model Dataloader"""

from __future__ import annotations
from typing import Any, List, Dict
from sklearn.utils import shuffle
from numpy.typing import NDArray
import numpy as np

from dataset import Dataset


class Dataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        truncate_excess: bool,
        do_shuffle: bool,
        *args,
        **kwargs,
    ) -> None:
        self.current_idx = 0
        self.dataset = dataset

        self.batch_size = batch_size
        self.truncate_excess = truncate_excess
        self.do_shuffle = do_shuffle

        self.org_list = shuffle(np.array(self.dataset.data_list))
        self.idx_list = np.arange(len(dataset))
        self.table = []

        self.reset()

        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.table):
            self.reset()
            return StopIteration

        batch = self.collect_batch(self.table[self.current_idx])
        self.current_idx += 1

        return batch

    def __len__(self):
        return len(self.table)

    def reset(self):
        self._init_table()
        self.current_idx = 0

    def _init_table(self):
        if self.do_shuffle:
            _data_list = shuffle(self.idx_list)
        else:
            _data_list = self.idx_list

        self.table = self.batching(_data_list)

    def batching(self, data_list: NDArray[Any]) -> List[NDArray[Any]]:
        batches = [[]]

        for i, data in enumerate(data_list):
            batches[-1].append(data)

            if i + 1 == len(data_list):
                if len(batches) > 1:
                    if len(batches[-1]) < self.batch_size and self.truncate_excess:
                        batches.pop(-1)
                break

            if len(batches[-1]) == self.batch_size:
                batches.append([])
                continue

        if batches == [[]]:
            batches = []

        return batches

    def collect_batch(self, batch_idx: List[int]) -> List[Any]:
        _batch = []
        for _idx in batch_idx:
            _batch.append(self.dataset[_idx])

        return _batch

    def train_valid(self, valid_rate: float, **_) -> Dict[str, Dataloader]:
        if not 0.0 < valid_rate < 1.0:
            raise ValueError("'valid_rate' must be (0.0, 1.0).")

        if self.do_shuffle:
            _idx_list = shuffle(self.idx_list)
        else:
            _idx_list = self.idx_list

        all_data = len(self.dataset)
        train_list = _idx_list[: int(-all_data * valid_rate)]
        valid_list = _idx_list[int(-all_data * valid_rate) :]

        train_loader = self.copy()
        valid_loader = self.copy()

        train_loader.idx_list = train_list
        valid_loader.idx_list = valid_list

        return {"train": train_loader.reset(), "valid": valid_loader.reset()}

    def copy(self) -> Dataloader:
        _copy = Dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            truncate_excess=self.truncate_excess,
            do_shuffle=self.do_shuffle,
            *self.args,
            **self.kwargs,
        )
        return _copy
