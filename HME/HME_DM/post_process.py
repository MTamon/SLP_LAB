"""This program is for shapping Head-Motion-Estimation results and make subset"""

from argparse import Namespace
from logging import Logger
import os
import re
from typing import Tuple, List
from modules import Collector, Condition, Directory
from modules import CEJC_Builder
from utils import get_post_args, get_extraction_args
from logger_gen import set_logger


def cejc_condition(path: str) -> bool:
    """For CEJC dataset's condition"""

    dir_path = os.path.dirname(path)
    fil_name = os.path.basename(path)
    member_list = os.listdir(dir_path)
    _m_l = []

    if not "_MIX" in fil_name:
        return True

    all_mix_flg = True
    for member in member_list:
        if not os.path.isfile(os.path.join(dir_path, member)):
            continue
        if member[-4:] != ".mp4":
            continue

        _m_l.append(member)
        if not "_MIX" in member:
            all_mix_flg = False

    return all_mix_flg


class PostProcess:
    """Shapping result of Head-Motion-Estimation and save shaped results."""

    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        self.target = args.original
        self.hme_result = args.hme_result
        self.hme_result_par = os.path.split(self.hme_result)[0]
        # self.output = args.output

        condition1 = Condition().specify_extention(["mp4"])
        condition1.add_exclude_dirc(["DONOT_USE"])
        condition1.add_exclude_filename(["_HD", "_SK"])
        condition1.add_condition_func(cejc_condition)

        condition2 = []
        condition2.append(
            Condition()
            .add_exclude_dirc(["DONOT_USE"])
            .specify_extention(["wav"])
            .add_exclude_filename(["IC0A", "IC0B", "IC0C"])
        )
        condition2.append(
            Condition()
            .add_exclude_dirc(["DONOT_USE"])
            .specify_extention(["csv"])
            .add_contain_filename(["luu"])
        )

        self.conditions = (condition1, condition2)

        self.collector = Collector(self.conditions[0], self.target)
        self.builder = CEJC_Builder(self.logger, self.args)

    def __call__(self):
        self.add_new_files()
        out_root_path, out_collector = self.get_outsite_collector()

        # generate Shaper inputs
        extractor_input = get_extraction_args(self.collector, self.hme_result_par)
        shaper_input = self.builder.get_shape_inputs(extractor_input)

        # generate MatchAV inputs
        cv_wav_set = self.get_matchav_inputs(out_collector, out_root_path, shaper_input)

        self.builder(shaper_input, cv_wav_set)

    def add_new_files(self):
        """add new datas to output site"""
        database = self.collector.get_directory_instance()
        database.incarnate(self.hme_result_par, self.conditions[1], self.logger.info)

    def get_outsite_collector(self) -> Tuple[str, Collector]:
        """get output site Collector instance"""
        database = self.collector.get_directory_instance()
        cloned_path = "/".join([self.hme_result_par, database.name])
        cloned_path = "/".join(cloned_path.split("\\"))
        clone_collector = Collector(self.conditions[1], cloned_path)
        _db = clone_collector.database.clone(self.conditions[1])
        clone_collector.database = _db

        return (cloned_path, clone_collector)

    def get_matchav_inputs(
        self, out_collector: Collector, out_root_path: str, shaper_input
    ) -> List[str]:
        """get MatchAV __call__()'s inputs"""
        csv_wav_set = []

        for sh_subset in shaper_input:
            hp_outsite = "/".join(sh_subset[0].split("\\"))
            hp_outsite = os.path.dirname(hp_outsite)

            sh_name = os.path.basename(sh_subset[2])
            data_lot = None
            if re.search(r"[a-z]", sh_name.split("_")[1]) is not None:
                data_lot = sh_name.split("_")[1][-1]

            # generate Directory __call__()'s input path
            search_path = self.get_dif_path(hp_outsite, out_root_path)

            # get target Directory instance
            out_dirc = out_collector.database(search_path)

            # shape filemember
            shaped_dicts = self.collect_belong_files(out_dirc, data_lot)

            csv_wav_set.append(shaped_dicts)

        return csv_wav_set

    def get_dif_path(self, site1: str, site2: str) -> str:
        """get difference file path route from head"""
        _site1 = os.path.abspath(site1).split(os.sep)
        _site2 = os.path.abspath(site2).split(os.sep)

        if len(_site1) < len(_site2):
            tmp = _site2
            _site2 = _site1
            _site1 = tmp

        serch_path = []
        for i, _route1 in enumerate(_site1):
            if i < len(_site2):
                assert _route1 == _site2[i]
                continue
            serch_path.append(_route1)

        if serch_path == []:
            return ""
        else:
            serch_path = "/".join(serch_path)
            return serch_path

    def collect_belong_files(self, out_dirc: Directory, data_lot: str) -> dict:
        """collect file member .csv and .wav, which match with data_lot"""
        shaped_dicts = {".csv": None, ".wav": []}
        for mem in out_dirc.file_member:
            # matching data-lot for example 'C002_006a_...' => data_lot is 'a'
            _data_lot = None
            mem_name = os.path.basename(mem)
            if re.search(r"[a-z]", re.split(r"[_-]", mem_name)[1]) is not None:
                _data_lot = re.split(r"[_-]", mem_name)[1][-1]
            if _data_lot != data_lot:
                continue

            res_dict = {}
            res_dict["path"] = mem
            res_dict["name"] = os.path.basename(mem)
            res_dict["extn"] = os.path.splitext(mem)[-1]
            res_dict["dlot"] = _data_lot

            if res_dict["extn"] == ".csv":
                shaped_dicts[".csv"] = res_dict
            elif res_dict["extn"] == ".wav":
                shaped_dicts[".wav"].append(res_dict)

        return shaped_dicts


if __name__ == "__main__":
    _args = get_post_args()
    _logger = set_logger("MAKE-SUBSET", _args.log)

    processor = PostProcess(_logger, _args)
    processor()
