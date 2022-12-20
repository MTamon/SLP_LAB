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
        _data_list = {}

        for fname in tqdm(self.data_list, desc="Rearangement-Table"):

            finfo = fname.split("_")
            ftype = "_".join(finfo[:4])
            fstrt = int(finfo[4])

            if ftype in _data_list:
                _data_list[ftype] = {}

            segment_path = self.datasite + "/" + fname
            with open(segment_path, "rb") as seg:
                segment = pickle.load(seg)
                vfps = segment["vfps"]
                ffps = segment["ffps"]

            for i in range(len(segment["cent"])):
                fframe = int(i / vfps * ffps)
                ac_prev_idx = fframe + 1 - self.acostic_frame_width
                ph_prev_idx = i + 1 - self.physic_frame_width

                if fframe + 1 == len(segment["trgt"]) or i + 1 == len(segment["cent"]):
                    break
                if ac_prev_idx < 1 or ph_prev_idx < 1:
                    continue

                abs_idx = i + fstrt
                key = self.make_dict_key(abs_idx)
                _set = [fname, i, abs_idx]
                if not self.binary_search(_data_list, ftype, key, abs_idx):
                    _data_list = self.insertion_sort(_data_list, ftype, key, _set)

        new_dl = []
        for _type in enumerate(tqdm(_data_list, desc="     Shaping-Table")):
            for _key in _data_list[_type]:
                new_dl += _data_list[_type][_key]

        return new_dl

    def binary_search(self, _data_list, ftype, key, abs_idx):
        target = _data_list[ftype][key]

        left_index = 0
        right_index = len(target) - 1
        while left_index <= right_index:
            middle_index = (left_index + right_index) // 2
            middle_value = target[middle_index][2]

            if abs_idx < middle_value:
                right_index = middle_index - 1
                continue
            if abs_idx > middle_value:
                left_index = middle_index + 1
                continue

            return True

        return False

    def make_dict_key(self, idx):
        s = idx // 100
        e = s + 1
        return (s * 100, e * 100)

    def insertion_sort(self, _data_list, ftype, key, _set):
        target: list = _data_list[ftype][key]

        if len(target) == 0:
            target = [_set]
            _data_list[ftype][key] = target
            return _data_list

        left_index = 0
        right_index = len(target) - 1
        insert_idx = -1
        not_found = True
        while left_index <= right_index:
            middle_index = (left_index + right_index) // 2
            middle_value = target[middle_index][2]

            if _set[2] < middle_value:
                right_index = middle_index - 1
                if right_index >= 0:
                    if target[right_index][2] < _set[2]:
                        insert_idx = right_index
                        break
                else:
                    insert_idx = 0
                continue
            if _set[2] > middle_value:
                left_index = middle_index + 1
                if left_index < len(target):
                    if target[left_index][2] > _set[2]:
                        insert_idx = left_index - 1
                        break
                else:
                    insert_idx = len(target)
                continue

            target.insert(middle_index, _set)
            not_found = False
            break

        if not_found:
            target.insert(insert_idx, _set)
        _data_list[ftype][key] = target
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
