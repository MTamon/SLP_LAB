from argparse import ArgumentParser, Namespace
import datetime
import re
import random
import torch
import numpy as np


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
        "--use-model",
        default="alpha",
        type=str,
        help="Model type (architecture). 'alpha' or 'alphaS'",
    )
    parser.add_argument(
        "--model-type",
        default="mediam",
        type=str,
        help="Model type (size). 'mediam' or 'small' or 'large'",
    )
    parser.add_argument(
        "--use-power",
        default=False,
        action="store_true",
        help="Parameter for using log-power.",
    )
    parser.add_argument(
        "--use-delta",
        default=False,
        action="store_true",
        help="Parameter for using delta feature.",
    )
    parser.add_argument(
        "--use-person1",
        default=False,
        action="store_true",
        help="Parameter for using person1's voice.",
    )
    parser.add_argument(
        "--use-person2",
        default=False,
        action="store_true",
        help="Parameter for using person2's voice.",
    )
    # parser.add_argument(
    #     "--acostic-dim",
    #     default=None,
    #     type=int,
    #     help="Acoustic feature frame size [dim]",
    # )
    # parser.add_argument(
    #     "--num-layers",
    #     default=None,
    #     type=int,
    #     help="ContextNet encoder parameter.",
    # )
    # parser.add_argument(
    #     "--kernel-size",
    #     default=None,
    #     type=int,
    #     help="ContextNet encoder parameter.",
    # )
    # parser.add_argument(
    #     "--out-kernel-size",
    #     default=None,
    #     type=int,
    #     help="Output convolusion layer's kernel size.",
    # )
    # parser.add_argument(
    #     "--num-channels",
    #     default=None,
    #     type=int,
    #     help="ContextNet encoder parameter.",
    # )
    # parser.add_argument(
    #     "--cnet-out-dim",
    #     default=None,
    #     type=int,
    #     help="ContextNet encoder parameter.",
    # )
    # parser.add_argument(
    #     "--encoder-dim",
    #     default=None,
    #     type=int,
    #     help="Encoder output dimension size.",
    # )
    # parser.add_argument(
    #     "--physic-frame-width",
    #     default=None,
    #     type=int,
    #     help="Acoustic feature frame num in input ContextNet [frame]",
    # )
    # parser.add_argument(
    #     "--acostic-frame-width",
    #     default=None,
    #     type=int,
    #     help="Acoustic feature frame num in input ContextNet [frame]",
    # )
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
        default=100,
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
    _args.use_person = (_args.use_person1, _args.use_person2)

    check_args(_args)

    return _args


def check_args(args: Namespace):
    if not args.model_type in ["mediam", "small", "large"]:
        raise ValueError(f"Invalid --use-mode value {args.model_type}")


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


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
