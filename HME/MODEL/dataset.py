"""For Simple-Model Dataset"""

from argparse import Namespace
from typing import List
import os
import re
import pickle
import numpy as np
import torch


class Dataset:
    def __init__(self, args: Namespace):
        self.args = args

        self.datasite = args.datasite
        self.ac_feature_size = args.ac_feature_size
        self.ac_feature_width = args.ac_feature_width

        self.data_list = os.listdir(self.datasite)
        self.current_idx = 0

        self.datasite = "/".join(re.split(r"[\\/]", self.datasite))

    def __len__(self):
        return len(self.data_list)

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

    def _get_data(self, file_name: str) -> List[torch.Tensor]:
        segment_path = self.datasite + "/" + file_name
        with open(segment_path, "rb") as seg:
            segment = pickle.load(seg)

        vfps = segment["vfps"]
        ffps = segment["ffps"]

        cent = torch.Tensor(segment["angl"])
        angl = torch.Tensor(segment["cent"])
        trgt = []
        othr = []

        for i in range(len(cent)):
            fframe = int(i / vfps * ffps)
            prev_idx = fframe + 1 - self.ac_feature_width

            trgt_window = segment["trgt"][max(prev_idx, 0) : fframe + 1]
            othr_window = segment["othr"][max(prev_idx, 0) : fframe + 1]

            if len(trgt_window) < self.ac_feature_width:
                dif = self.ac_feature_width - len(trgt_window)
                pad = np.zeros((dif * self.ac_feature_size), dtype=trgt_window.dtype)

                trgt_window = np.concatenate((pad, trgt_window), axis=0)
                othr_window = np.concatenate((pad, othr_window), axis=0)

            trgt.append(trgt_window)
            othr.append(othr_window)

        trgt = torch.Tensor(np.stack(trgt))
        othr = torch.Tensor(np.stack(othr))

        return [angl, cent, trgt, othr]
