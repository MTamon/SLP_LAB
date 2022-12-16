"""Utils"""
from argparse import ArgumentParser, Namespace


def batching(data_list: list, batch_size: int) -> list:
    batches = [[]]

    for i, data in enumerate(data_list):
        batches[-1].append(data)

        if i + 1 == len(data_list):
            break

        if len(batches[-1]) == batch_size:
            batches.append([])
            continue

    if batches == [[]]:
        batches = []

    return batches


def add_mfcc_args(parser: ArgumentParser) -> ArgumentParser:
    """add argument"""

    parser.add_argument(
        "--origin",
        default=None,
        type=str,
        help="Path for original database.",
    )
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
    parser.add_argument(
        "--segment-size",
        default=15,
        type=float,
        help="Segment window size [s]. Defaults to 15",
    )
    parser.add_argument(
        "--segment-stride",
        default=5,
        type=float,
        help="Segment window stride [s]. Defaults to 5",
    )
    parser.add_argument(
        "--segment-min-size",
        default=10,
        type=float,
        help="Segment min size [s]. Defaults to 10",
    )
    parser.add_argument(
        "--sep_data",
        default=False,
        action="store_true",
        help="Excute process with single thread. Defaults to False.",
    )
    parser.add_argument(
        "--use-feature",
        default="mfcc",
        type=str,
        help="Acoustic feature computed by 'mfcc' or 'fbank'. Defaults to mfcc.",
    )
    parser.add_argument(
        "--video-fps",
        default=29.97,
        type=float,
        help="Segment min size [s]. Defaults to 10",
    )
    parser.add_argument(
        "--sample-frequency",
        default=16000,
        type=float,
        help="Sampling frequency of input waveform [Hz]. Defaults to 16000.",
    )
    parser.add_argument(
        "--frame-length",
        default=32,
        type=int,
        help="frame-size [ms]. Defaults to 25.",
    )
    parser.add_argument(
        "--frame-shift",
        default=13,
        type=int,
        help="Analysis interval (frame shift) [ms]. Defaults to 10.",
    )
    parser.add_argument(
        "--num-mel-bins",
        default=23,
        type=int,
        help="Number of mel filter banks (= number of dimensions of FBANK features). Defaults to 23.",
    )
    parser.add_argument(
        "--num-ceps",
        default=13,
        type=int,
        help="Number of dimensions of MFCC features (including the 0th dimension). Defaults to 13.",
    )
    parser.add_argument(
        "--low-frequency",
        default=20,
        type=float,
        help="Cutoff frequency for low frequency band rejection. Defaults to 20.",
    )
    parser.add_argument(
        "--high-frequency",
        default=8000,
        type=float,
        help="Cutoff frequency for high frequency band rejection. Defaults to 8000.",
    )
    parser.add_argument(
        "--dither",
        default=1.0,
        type=float,
        help="Dithering process parameters (noise strength). Defaults to 1.0.",
    )
    parser.add_argument(
        "--proc-num",
        default=7,
        type=int,
        help="Multiprocessing thread number. Defaults to 7.",
    )
    parser.add_argument(
        "--single_proc",
        default=False,
        action="store_true",
        help="Excute process with single thread. Defaults to False.",
    )
    parser.add_argument(
        "--redo",
        default=False,
        action="store_true",
        help="Redo process.",
    )
    parser.add_argument(
        "--convert-path",
        default=None,
        type=int,
        help="Specially prepared for this experiment.",
    )

    return parser


def get_mfcc_args() -> Namespace:
    """generate ArgumentParser instance."""
    parser = ArgumentParser("This program is for head-motion-estimation analisis.")
    parser = add_mfcc_args(parser)

    return parser.parse_args()
