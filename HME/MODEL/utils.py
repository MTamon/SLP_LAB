from argparse import ArgumentParser, Namespace
import datetime
import re


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
        default="log/train.log",
        type=str,
        help="Path for log-file.",
    )
    parser.add_argument(
        "--train-result-path",
        default="transition/train.csv",
        type=str,
        help="Path for result-file.",
    )
    parser.add_argument(
        "--valid-result-path",
        default="transition/valid.csv",
        type=str,
        help="Path for result-file.",
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
        "--lstm-input-dim",
        default=128,
        type=int,
        help="LSTM forward Linear size [dim]",
    )
    parser.add_argument(
        "--lstm-output-dim",
        default=128,
        type=int,
        help="LSTM backward Linear size [dim]",
    )
    parser.add_argument(
        "--num-layer",
        default=1,
        type=int,
        help="LSTM layer num.",
    )
    parser.add_argument(
        "--ac-linear-dim",
        default=10,
        type=int,
        help="Number of dimensions of acoustic features after compression [dim]",
    )
    parser.add_argument(
        "--pos-feature-size",
        default=8,
        type=int,
        help="Number of dimensions of position features after compression [dim]",
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
        "--do-shuffle",
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
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        help="One of beta value of AdamW",
    )
    parser.add_argument(
        "--beta2",
        default=0.98,
        type=float,
        help="One of beta value of AdamW",
    )
    parser.add_argument(
        "--eps",
        default=1e-8,
        type=float,
        help="eps which is parameter of optimizer AdamW",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-2,
        type=float,
        help="weight decay coefficient of AdamW",
    )

    return parser


def get_args() -> Namespace:
    """generate ArgumentParser instance."""

    parser = ArgumentParser(
        "This program is for head-motion-estimation model learning."
    )
    parser = add_args(parser)

    _args = parser.parse_args()
    _args.betas = (_args.beta1, _args.beta2)

    return _args


def add_datetime_path(path: str):
    dt_now = str(datetime.datetime.now())
    dt_now = re.split(r"[.]", dt_now)[0]
    dt_now = "".join(re.split(r"[-:]", dt_now))
    dt_now = "_".join(re.split(r"[ ]", dt_now))

    _path = re.split(r"[.]", path)
    _path[-2] += "_" + dt_now
    path = ".".join(_path)

    return path


def override(klass):
    def check_super(method):
        method_name = method.__name__
        msg = f"`{method_name}()` is not defined in `{klass.__name__}`."
        assert method_name in dir(klass), msg

    def wrapper(method):
        check_super(method)
        return method

    return wrapper
