"""Utils"""

from argparse import ArgumentParser
import os
from modules import Collector
from modules import add_fm_args, add_build_args


def add_hme_args(parser: ArgumentParser) -> ArgumentParser:
    """add argument"""
    parser = add_fm_args(parser)

    parser.add_argument(
        "--target",
        default="./",
        type=str,
        help="Path for target database.",
    )
    parser.add_argument(
        "--output",
        default="out",
        type=str,
        help="Path for output results.",
    )
    parser.add_argument(
        "--log",
        default="log",
        type=str,
        help="Path for log-file.",
    )

    return parser


def get_hme_args():
    """generate ArgumentParser instance."""
    parser = ArgumentParser("This program is for head-motion-estimation analisis.")
    parser = add_hme_args(parser)

    return parser.parse_args()


def get_extraction_args(
    collector: Collector, output_path: str, extention: str = ".mp4"
):
    """make Extraction callable instance's arguments"""
    file_list = collector.get_path()
    file_list = collector.serialize_path_list(file_list)

    if output_path[-1] == "/":
        output_path = output_path[:-1]
    output_path = "/".join(output_path.split("\\"))

    database = collector.get_directory_instance()
    db_abspath_len = len(database.get_abspath().split("/"))
    root_path = "/".join([output_path, database.name])

    ex_arg = []
    for file in file_list:
        _file = os.path.abspath(file).split(os.sep)
        under_path = "/".join(_file[db_abspath_len:])
        under_path = under_path[: -len(extention)]

        hpe_path = "/".join([root_path, under_path + ".hp"])
        trim_path = "/".join([root_path, under_path + ".area"])

        ex_arg.append((file, hpe_path, trim_path, None, None))

    return ex_arg


def add_post_args(parser: ArgumentParser) -> ArgumentParser:
    """add argument"""
    parser = add_build_args(parser)

    parser.add_argument(
        "--original",
        default="./",
        type=str,
        help="Path for original database.",
    )
    parser.add_argument(
        "--hme-result",
        default="./",
        type=str,
        help="Path for hme results.",
    )
    # parser.add_argument(
    #     "--output",
    #     default="out",
    #     type=str,
    #     help="Path for output results.",
    # )
    parser.add_argument(
        "--log",
        default="log",
        type=str,
        help="Path for log-file.",
    )

    return parser


def get_post_args():
    """generate ArgumentParser instance."""
    parser = ArgumentParser(
        "This program is for shapping result of head-motion-estimation."
    )
    parser = add_post_args(parser)

    return parser.parse_args()
