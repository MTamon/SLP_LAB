"""For Simple-Model Dataloader"""

from typing import Any, List
from sklearn.utils import shuffle
from numpy.typing import NDArray
import numpy as np

from dataset import Dataset


class Dataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        valid_rate: float,
        truncate_excess: bool,
        do_shuffle: bool,
        *args,
        **kwargs,
    ) -> None:
        self.current_idx = 0
        self.dataset = dataset

        self.batch_size = batch_size
        self.valid_rate = valid_rate
        self.truncate_excess = truncate_excess
        self.do_shuffle = do_shuffle

        _org_list = shuffle(np.array(self.dataset.data_list))
        self.train_list = _org_list[: int(-len(_org_list) * self.valid_rate)]
        self.valid_list = _org_list[int(-len(_org_list) * self.valid_list) :]
        self.table = []

        self._init_table()

        self.args = args
        self.kwargs = kwargs

    def __next__(self):
        if self.current_idx >= len(self.table):
            self._init_table()
            return StopIteration

        batch = self.table[self.current_idx]
        self.current_idx += 1

        return batch

    def __len__(self):
        return len(self.table)

    def _reset(self):
        self.current_idx = 0

    def _init_table(self):
        if self.do_shuffle:
            _data_list = shuffle(self.train_list)
        else:
            _data_list = self.train_list

        self.table = self.batching(_data_list)
        self._reset()

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
