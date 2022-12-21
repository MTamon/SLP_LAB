"""For Simple-Model Dataloader"""

from __future__ import annotations
from typing import List, Dict
import torch
import joblib

from src import override
from src import Dataloader
from ohta_dataset import OhtaDataset


class OhtaDataloader(Dataloader):
    def __init__(
        self,
        dataset: OhtaDataset,
        batch_size: int,
        truncate_excess: bool,
        do_shuffle: bool,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset, batch_size, truncate_excess, do_shuffle, *args, **kwargs
        )
        self.args = args
        self.kwargs = kwargs

    @override(Dataloader)
    def collect_batch(
        self, batch_idx: List[int]
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        record = self.dataset[batch_idx[0]]
        _batch = record

        result = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.thread)(i) for i in batch_idx[1:]
        )

        for record in result:
            _batch[0][0] += record[0][0]
            _batch[0][1] += record[0][1]
            _batch[0][2] += record[0][2]
            _batch[0][3] += record[0][3]
            _batch[0][4] += record[0][4]
            _batch[0][5] += record[0][5]
            _batch[0][6] += record[0][6]
            _batch[0][7] += record[0][7]
            _batch[0][8] += record[0][8]
            _batch[0][9] += record[0][9]
            _batch[0][10] += record[0][10]
            _batch[0][11] += record[0][11]
            _batch[1][0] += record[1][0]
            _batch[1][1] += record[1][1]

        try:
            batch = [
                [
                    [torch.stack(_batch[0][0]), torch.stack(_batch[0][1])],
                    [torch.stack(_batch[0][2]), torch.stack(_batch[0][3])],
                    [torch.stack(_batch[0][4]), torch.stack(_batch[0][5])],
                    [torch.stack(_batch[0][6]), torch.stack(_batch[0][7])],
                    [torch.stack(_batch[0][8]), torch.stack(_batch[0][9])],
                    [torch.stack(_batch[0][10]), torch.stack(_batch[0][11])],
                ],
                [
                    torch.stack(_batch[1][0]),
                    torch.stack(_batch[1][1]),
                ],
            ]
        except Exception as exc:
            print(record[2])
            raise exc

        return batch

    def thread(self, idx):
        return self.dataset[idx]

    @override(Dataloader)
    def copy(self) -> OhtaDataloader:
        _copy = OhtaDataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            truncate_excess=self.truncate_excess,
            do_shuffle=self.do_shuffle,
            *self.args,
            **self.kwargs,
        )
        return _copy
