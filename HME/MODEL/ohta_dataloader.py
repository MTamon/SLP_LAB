"""For Simple-Model Dataloader"""

from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import joblib
import pickle

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
        self.dataset = dataset
        self.args = args
        self.kwargs = kwargs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @override(Dataloader)
    def collect_batch(
        self, batch_idx: List[int]
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        record = OhtaDataloader.get_data(*self._get_args(0))
        _batch = record

        result = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(OhtaDataloader.get_data)(*self._get_args(i))
            for i in batch_idx[1:]
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
                    [
                        torch.stack(_batch[0][0]).to(device=self.device),
                        torch.stack(_batch[0][1]).to(device=self.device),
                    ],
                    [
                        torch.stack(_batch[0][2]).to(device=self.device),
                        torch.stack(_batch[0][3]).to(device=self.device),
                    ],
                    [
                        torch.stack(_batch[0][4]).to(device=self.device),
                        torch.stack(_batch[0][5]).to(device=self.device),
                    ],
                    [
                        torch.stack(_batch[0][6]).to(device=self.device),
                        torch.stack(_batch[0][7]).to(device=self.device),
                    ],
                    [
                        torch.stack(_batch[0][8]).to(device=self.device),
                        torch.stack(_batch[0][9]).to(device=self.device),
                    ],
                    [
                        torch.stack(_batch[0][10]).to(device=self.device),
                        torch.stack(_batch[0][11]).to(device=self.device),
                    ],
                ],
                [
                    torch.stack(_batch[1][0]).to(device=self.device),
                    torch.stack(_batch[1][1]).to(device=self.device),
                ],
            ]
        except Exception as exc:
            print(record[2])
            raise exc

        return batch

    @staticmethod
    def get_data(
        datasite: str,
        file_pointer: Tuple[str, int],
        acostic_frame_width: int,
        physics_frame_width: int,
        acostic_dim: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        segment_path = datasite + "/" + file_pointer[0]
        with open(segment_path, "rb") as seg:
            segment = pickle.load(seg)

        vfps = segment["vfps"]
        ffps = segment["ffps"]

        common = {
            "device": torch.device("cpu"),
            "dtype": torch.float32,
            "requires_grad": True,
        }

        i = file_pointer[1]

        fframe = int(i / vfps * ffps)
        ac_prev_idx = fframe + 1 - acostic_frame_width
        ph_prev_idx = i + 1 - physics_frame_width

        trgt_window = torch.tensor(segment["trgt"][ac_prev_idx : fframe + 1], **common)
        othr_window = torch.tensor(segment["othr"][ac_prev_idx : fframe + 1], **common)
        trgt_lpower = torch.tensor(segment["tlgp"][ac_prev_idx : fframe + 1], **common)
        othr_lpower = torch.tensor(segment["olgp"][ac_prev_idx : fframe + 1], **common)

        assert trgt_window.shape[0] == acostic_frame_width
        assert othr_window.shape[0] == acostic_frame_width
        assert trgt_window.shape[1] == acostic_dim
        assert othr_window.shape[1] == acostic_dim
        assert len(trgt_lpower.shape) == 1, f"FILE : {file_pointer[0]}"
        assert len(othr_lpower.shape) == 1, f"FILE : {file_pointer[0]}"

        _trgt_window = torch.tensor(segment["trgt"][ac_prev_idx - 1 : fframe], **common)
        _othr_window = torch.tensor(segment["othr"][ac_prev_idx - 1 : fframe], **common)
        dtrgt_window = OhtaDataloader.clone_detach(trgt_window - _trgt_window)
        dothr_window = OhtaDataloader.clone_detach(othr_window - _othr_window)

        _trgt_lpower = torch.tensor(segment["tlgp"][ac_prev_idx - 1 : fframe], **common)
        _othr_lpower = torch.tensor(segment["olgp"][ac_prev_idx - 1 : fframe], **common)
        dtrgt_lpower = OhtaDataloader.clone_detach(trgt_lpower - _trgt_lpower)
        dothr_lpower = OhtaDataloader.clone_detach(othr_lpower - _othr_lpower)

        cent_window = torch.tensor(segment["cent"][ph_prev_idx : i + 1], **common)
        angl_window = torch.tensor(segment["angl"][ph_prev_idx : i + 1], **common)

        _cent_window = torch.tensor(segment["cent"][ph_prev_idx - 1 : i], **common)
        _angl_window = torch.tensor(segment["angl"][ph_prev_idx - 1 : i], **common)
        dcent_window = OhtaDataloader.clone_detach(cent_window - _cent_window)
        dangl_window = OhtaDataloader.clone_detach(angl_window - _angl_window)

        ans_angl = segment["angl"][i + 1] - segment["angl"][i]
        ans_cent = segment["cent"][i + 1] - segment["cent"][i]

        one_set = (
            [
                [angl_window],
                [dangl_window],
                [cent_window],
                [dcent_window],
                [trgt_window],
                [dtrgt_window],
                [othr_window],
                [dothr_window],
                [trgt_lpower],
                [dtrgt_lpower],
                [othr_lpower],
                [dothr_lpower],
            ],
            [
                [torch.tensor(ans_angl, **common)],
                [torch.tensor(ans_cent, **common)],
            ],
            file_pointer[0],
        )

        return one_set

    @staticmethod
    def clone_detach(tensor: torch.Tensor):
        return tensor.clone().detach().requires_grad_(True)

    def _get_args(self, idx):
        return (
            self.dataset.datasite,
            self.dataset[idx],
            self.dataset.acostic_frame_width,
            self.dataset.physic_frame_width,
            self.dataset.acostic_dim,
        )

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
