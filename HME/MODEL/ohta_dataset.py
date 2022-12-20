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

        self.data_list = self.rearange_table()

    def rearange_table(self):
        _data_list = []

        for fname in tqdm(self.data_list, desc="Rearangement-Table"):
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

        common = {"device": self.device, "dtype": torch.float32}

        i = file_pointer[1]

        fframe = int(i / vfps * ffps)
        ac_prev_idx = fframe + 1 - self.acostic_frame_width
        ph_prev_idx = i + 1 - self.physic_frame_width

        trgt_window = torch.tensor(segment["trgt"][ac_prev_idx : fframe + 1], **common)
        othr_window = torch.tensor(segment["othr"][ac_prev_idx : fframe + 1], **common)
        trgt_lpower = torch.tensor(segment["tlgp"][ac_prev_idx : fframe + 1], **common)
        othr_lpower = torch.tensor(segment["olgp"][ac_prev_idx : fframe + 1], **common)

        _trgt_window = segment["trgt"][ac_prev_idx - 1 : fframe]
        _othr_window = segment["othr"][ac_prev_idx - 1 : fframe]
        dtrgt_window = torch.tensor(trgt_window - _trgt_window, **common)
        dothr_window = torch.tensor(othr_window - _othr_window, **common)

        _trgt_lpower = segment["tlgp"][ac_prev_idx - 1 : fframe]
        _othr_lpower = segment["olgp"][ac_prev_idx - 1 : fframe]
        dtrgt_lpower = torch.tensor(trgt_lpower - _trgt_lpower, **common)
        dothr_lpower = torch.tensor(othr_lpower - _othr_lpower, **common)

        cent_window = torch.tensor(segment["cent"][ph_prev_idx : i + 1], **common)
        angl_window = torch.tensor(segment["angl"][ph_prev_idx : i + 1], **common)

        _cent_window = segment["cent"][ph_prev_idx - 1 : i]
        _angl_window = segment["angl"][ph_prev_idx - 1 : i]
        dcent_window = torch.tensor(cent_window - _cent_window, **common)
        dangl_window = torch.tensor(angl_window - _angl_window, **common)

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
