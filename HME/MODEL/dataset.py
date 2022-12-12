"""For Simple-Model Dataset"""

from abc import ABCMeta, abstractmethod
import os
import re


class Dataset(metaclass=ABCMeta):
    def __init__(self, datasite, *args, **kwargs):

        self.datasite = datasite
        self.args = args
        self.kwargs = kwargs

        self.data_list = os.listdir(self.datasite)
        self.current_idx = 0

        self.datasite = "/".join(re.split(r"[\\/]", self.datasite))

    def __len__(self):
        return len(self.data_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.data_list):
            self._reset()
            raise StopIteration

        _data = self._get_data(self.data_list[self.current_idx])
        self.current_idx += 1

        return _data

    def __getitem__(self, idx):
        return self._get_data(self.data_list[idx])

    def _reset(self):
        self.current_idx = 0

    @abstractmethod
    def _get_data(self, file_name: str):
        pass
