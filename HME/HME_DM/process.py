"""This program processing Head-Motion-Estimation analisis"""

from argparse import Namespace
from logging import Logger
import os
from modules import Collector, Condition
from modules import Extraction
from utils import get_hme_args, get_extraction_args
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


class MakeData:
    """Analize MP4 data with Head-Motion-Estimation and save analisis results."""

    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        self.target = args.target
        self.output = args.output

        condition1 = Condition().specify_extention(["mp4"])
        condition1.add_exclude_dirc(["DONOT_USE"])
        condition1.add_exclude_filename(["_HD", "_SK"])
        condition1.add_condition_func(cejc_condition)

        condition2 = Condition().specify_extention(["wav"])
        condition2.add_exclude_dirc(["DONOT_USE"])

        self.conditions = (condition1, condition2)

        self.collector = Collector(self.conditions[0], self.target)
        self.extractor = Extraction(logger, args)

    def __call__(self):
        self.make_output_site()

        extractor_input = get_extraction_args(self.collector, self.output)
        result = self.extractor(extractor_input)

        return result

    def make_output_site(self):
        """generate output directory structures"""
        database = self.collector.get_directory_instance()
        database.incarnate(self.output, self.conditions[1], self.logger.info)


if __name__ == "__main__":
    _args = get_hme_args()
    _logger = set_logger("DATA_MAKE", _args.log)

    processor = MakeData(_logger, _args)
    processor()
