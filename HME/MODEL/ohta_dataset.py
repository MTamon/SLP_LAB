"""For Simple-Model Dataset"""

from typing import List, Tuple
import pickle
import torch
from tqdm import tqdm

from src import Dataset


class OhtaDataset(Dataset):
    def __init__(
        self,
        datasite: str,
        *args,
        acostic_frame_width: int = 69,
        acostic_dim: int = 80,
        physic_frame_width: int = 10,
        **kwargs
    ):
        super().__init__(datasite, *args, **kwargs)

        self.acostic_frame_width = acostic_frame_width
        self.acostic_dim = acostic_dim
        self.physic_frame_width = physic_frame_width

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._conf_dict = {}
        self.data_list = self.rearange_table()

    def rearange_table(self):
        _data_list = []

        for fname in tqdm(self.data_list, desc="Rearangement-Table"):

            finfo = fname.split("_")
            ftype = "_".join(finfo[:4])
            fstrt = int(finfo[4])
            if not ftype in self._conf_dict.keys():
                self._conf_dict[ftype] = []

            segment_path = self.datasite + "/" + fname
            with open(segment_path, "rb") as seg:
                segment = pickle.load(seg)
                vfps = segment["vfps"]
                ffps = segment["ffps"]

            for i in range(len(segment["cent"])):
                if (i + fstrt) in self._conf_dict[ftype]:
                    continue
                else:
                    self._conf_dict[ftype].append(i + fstrt)

                fframe = int(i / vfps * ffps)
                ac_prev_idx = fframe + 1 - self.acostic_frame_width
                ph_prev_idx = i + 1 - self.physic_frame_width

                if fframe + 1 == len(segment["trgt"]) or i + 1 == len(segment["cent"]):
                    break
                if ac_prev_idx < 1 or ph_prev_idx < 1:
                    continue

                _data_list.append((fname, i))

        return _data_list

    def _get_data(
        self, file_pointer: Tuple[str, int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        segment_path = self.datasite + "/" + file_pointer[0]
        with open(segment_path, "rb") as seg:
            segment = pickle.load(seg)

        vfps = segment["vfps"]
        ffps = segment["ffps"]

        common = {"device": self.device, "dtype": torch.float32, "requires_grad": True}

        i = file_pointer[1]

        fframe = int(i / vfps * ffps)
        ac_prev_idx = fframe + 1 - self.acostic_frame_width
        ph_prev_idx = i + 1 - self.physic_frame_width

        trgt_window = torch.tensor(segment["trgt"][ac_prev_idx : fframe + 1], **common)
        othr_window = torch.tensor(segment["othr"][ac_prev_idx : fframe + 1], **common)
        trgt_lpower = torch.tensor(segment["tlgp"][ac_prev_idx : fframe + 1], **common)
        othr_lpower = torch.tensor(segment["olgp"][ac_prev_idx : fframe + 1], **common)

        _trgt_window = torch.tensor(segment["trgt"][ac_prev_idx - 1 : fframe], **common)
        _othr_window = torch.tensor(segment["othr"][ac_prev_idx - 1 : fframe], **common)
        dtrgt_window = self.clone_detach(trgt_window - _trgt_window)
        dothr_window = self.clone_detach(othr_window - _othr_window)

        _trgt_lpower = torch.tensor(segment["tlgp"][ac_prev_idx - 1 : fframe], **common)
        _othr_lpower = torch.tensor(segment["olgp"][ac_prev_idx - 1 : fframe], **common)
        dtrgt_lpower = self.clone_detach(trgt_lpower - _trgt_lpower)
        dothr_lpower = self.clone_detach(othr_lpower - _othr_lpower)

        cent_window = torch.tensor(segment["cent"][ph_prev_idx : i + 1], **common)
        angl_window = torch.tensor(segment["angl"][ph_prev_idx : i + 1], **common)

        _cent_window = torch.tensor(segment["cent"][ph_prev_idx - 1 : i], **common)
        _angl_window = torch.tensor(segment["angl"][ph_prev_idx - 1 : i], **common)
        dcent_window = self.clone_detach(cent_window - _cent_window)
        dangl_window = self.clone_detach(angl_window - _angl_window)

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
        )

        return one_set

    def clone_detach(self, tensor: torch.Tensor):
        return tensor.clone().detach().requires_grad_(True)
