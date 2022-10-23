"""This program processing Head-Motion-Estimation analisis"""

from argparse import Namespace
from logging import Logger
from modules import Collector, Condition
from modules import Extraction
from utils import get_args, get_extraction_args
from logger_gen import set_logger


class MakeData:
    """Analize MP4 data with Head-Motion-Estimation and save analisis results."""

    def __init__(self, logger: Logger, args: Namespace):
        self.logger = logger
        self.args = args

        self.target = args.target
        self.output = args.output

        self.conditions = (
            Condition().specify_extention("mp4"),
            Condition().specify_extention("wav"),
        )

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
    _args = get_args()
    _logger = set_logger("DATA_MAKE", _args.log)

    processor = MakeData(_logger, _args)
    processor()
