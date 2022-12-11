"""For Simple-Model Dataloader"""

from argparse import Namespace
from sklearn.utils import shuffle
from typing import Any, List
from numpy.typing import NDArray
import numpy as np

from dataset import Dataset


class Dataloader:
    def __init__(self, argparse: Namespace, dataset: Dataset) -> None:
        self.current_idx = 0

        self.argparse = argparse
        self.dataset = dataset

        self.batch_size = argparse.batch_size
        self.truncate_excess = argparse.truncate_excess
        self.shuffle = argparse.shuffle

        self.original_list = np.array(self.dataset.data_list)
        self.table = []

        self._init_table()

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
        if self.shuffle:
            _data_list = shuffle(self.original_list)
        else:
            _data_list = self.original_list

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
