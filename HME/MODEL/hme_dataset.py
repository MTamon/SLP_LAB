"""For Simple-Model Dataset"""

from typing import List, Tuple
import pickle
import numpy as np
import torch

from src import Dataset


class HmeDataset(Dataset):
    def __init__(
        self,
        datasite: str,
        ac_feature_size: int,
        ac_feature_width: int,
        *args,
        **kwargs
    ):
        super().__init__(datasite, *args, **kwargs)

        self.ac_feature_size = ac_feature_size
        self.ac_feature_width = ac_feature_width

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _get_data(
        self, file_name: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        segment_path = self.datasite + "/" + file_name
        with open(segment_path, "rb") as seg:
            segment = pickle.load(seg)

        vfps = segment["vfps"]
        ffps = segment["ffps"]

        angl = torch.tensor(segment["angl"], device=self.device)
        cent = torch.tensor(segment["cent"], device=self.device)
        trgt = []
        othr = []

        for i in range(len(cent)):
            fframe = int(i / vfps * ffps)
            prev_idx = fframe + 1 - self.ac_feature_width

            trgt_window = segment["trgt"][max(prev_idx, 0) : fframe + 1].flatten()
            othr_window = segment["othr"][max(prev_idx, 0) : fframe + 1].flatten()

            if len(trgt_window) < self.ac_feature_width * self.ac_feature_size:
                dif = self.ac_feature_width - len(trgt_window) // self.ac_feature_size
                pad = np.zeros((dif * self.ac_feature_size), dtype=trgt_window.dtype)

                trgt_window = np.concatenate((pad, trgt_window), axis=0)
                othr_window = np.concatenate((pad, othr_window), axis=0)

            trgt.append(trgt_window)
            othr.append(othr_window)

        trgt = torch.tensor(np.stack(trgt), device=self.device)
        othr = torch.tensor(np.stack(othr), device=self.device)

        ans_cent = cent[1:].clone()
        cent = cent[:-1]

        ans_angl = angl[1:].clone()
        angl = angl[:-1]

        trgt = trgt[:-1].clone()
        othr = othr[:-1].clone()

        return ([angl, cent, trgt, othr], [ans_angl, ans_cent])
