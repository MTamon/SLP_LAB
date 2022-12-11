from argparse import ArgumentParser, Namespace


def add_args(parser: ArgumentParser) -> ArgumentParser:
    """add argument"""

    parser.add_argument(
        "--datasite",
        default=None,
        type=str,
        help="Path to the site which stor data.",
    )
    parser.add_argument(
        "--log",
        default="log",
        type=str,
        help="Path for log-file.",
    )
    parser.add_argument(
        "--ac-feature-size",
        default=80,
        type=int,
        help="Acoustic feature frame size [dim]",
    )
    parser.add_argument(
        "--ac-feature-width",
        default=32,
        type=int,
        help="Acoustic feature frame num in input FC-layer [frame]",
    )
    parser.add_argument(
        "--lstm-dim",
        default=128,
        type=int,
        help="LSTM expression size [dim]",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Learning batch size.",
    )
    parser.add_argument(
        "--valid-rate",
        default=0.3,
        type=float,
        help="Valid data rate in data divide.",
    )
    parser.add_argument(
        "--truncate-excess",
        default=False,
        action="store_true",
        help="Truncation of batch separate excess.",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Shuffle dataset table.",
    )
    parser.add_argument(
        "--epoch",
        default=20,
        type=int,
        help="Epoch num.",
    )
    parser.add_argument(
        "--model-save-path",
        default="model/model.pth",
        type=str,
        help="Path for model saved site.",
    )
    parser.add_argument(
        "--save-per-epoch",
        default=False,
        action="store_true",
        help="Save model per epoch.",
    )

    return parser


def get_args() -> Namespace:
    """generate ArgumentParser instance."""

    parser = ArgumentParser(
        "This program is for head-motion-estimation model learning."
    )
    parser = add_args(parser)

    return parser.parse_args()
