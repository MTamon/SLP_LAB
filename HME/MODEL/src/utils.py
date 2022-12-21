from argparse import ArgumentParser, Namespace
from typing import List, Tuple
import pickle
import math
import datetime
import re
import os
import random
from tqdm import tqdm
import cv2
import torch
import numpy as np

from torch.optim import Optimizer
from torch.optim import AdamW
from model.ohta.modules.lamb import Lamb


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
        default=10,
        type=int,
        help="Number of dimensions of position features after compression [dim]",
    )
    parser.add_argument(
        "--relu-dim",
        default=10,
        type=int,
        help="Number of dimensions of Relu layer [dim]",
    )
    parser.add_argument(
        "--dropout-rate",
        default=0.5,
        type=float,
        help="Dropout rate of ReluUsed [dim]",
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
    parser.add_argument(
        "--use-model",
        default="simple",
        type=str,
        help="Use model.",
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

    check_args(_args)

    return _args


def choice_optimizer(
    params,
    use_optimizer="adamw",
    lr=1e-4,
    eps=1e-6,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    **_,
) -> Optimizer:
    if use_optimizer == "adamw":
        return AdamW(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif use_optimizer == "lamb":
        return Lamb(params=params)
    else:
        raise ValueError(f"Invalid optimizer name {use_optimizer}")


def check_args(args: Namespace):
    if not args.use_model in ["simple", "relu-used", "small"]:
        ValueError(f"Invalid --use-mode value {args.use_model}")


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


def rotation_matrix(
    theta1: float, theta2: float, theta3: float, order="xyz"
) -> np.ndarray:
    """
    入力
        theta1, theta2, theta3 = Angle of rotation theta 1, 2, 3 in order of rotation
        oreder = Order of rotation e.g. 'xzy' for X, Z, Y order
    出力
        3x3 Rotation Matrix
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == "xzx":
        matrix = np.array(
            [
                [c2, -c3 * s2, s2 * s3],
                [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "xyx":
        matrix = np.array(
            [
                [c2, s2 * s3, c3 * s2],
                [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yxy":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                [s2 * s3, c2, -c3 * s2],
                [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yzy":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                [c3 * s2, c2, s2 * s3],
                [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "zyz":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                [-c3 * s2, s2 * s3, c2],
            ]
        )
    elif order == "zxz":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                [s2 * s3, c3 * s2, c2],
            ]
        )
    elif order == "xyz":
        matrix = np.array(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
            ]
        )
    elif order == "xzy":
        matrix = np.array(
            [
                [c2 * c3, -s2, c2 * s3],
                [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
            ]
        )
    elif order == "yxz":
        matrix = np.array(
            [
                [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                [c2 * s3, c2 * c3, -s2],
                [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
            ]
        )
    elif order == "yzx":
        matrix = np.array(
            [
                [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                [s2, c2 * c3, -c2 * s3],
                [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
            ]
        )
    elif order == "zyx":
        matrix = np.array(
            [
                [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                [-s2, c2 * s3, c2 * c3],
            ]
        )
    elif order == "zxy":
        matrix = np.array(
            [
                [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                [-c2 * s3, s2, c2 * c3],
            ]
        )

    return matrix


def rotation_angles(matrix: np.ndarray, order: str = "xyz") -> np.ndarray:
    """
    Parameters
        matrix = 3x3 Rotation Matrix
        oreder = Order of rotation e.g. 'xzy' for X, Z, Y order
    Outputs
        theta1, theta2, theta3 = Angle of rotation theta 1, 2, 3 in order of rotation
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == "xzx":
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == "xyx":
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == "yxy":
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == "yzy":
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == "zyz":
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == "zxz":
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == "xzy":
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == "xyz":
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == "yxz":
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == "yzx":
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == "zyx":
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == "zxy":
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return np.array((theta1, theta2, theta3))


def get_data(
    segment_path: str, ac_feature_width, ac_feature_size
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    with open(segment_path, "rb") as seg:
        segment = pickle.load(seg)

    vfps = segment["vfps"]
    ffps = segment["ffps"]

    angl = torch.tensor(segment["angl"])
    cent = torch.tensor(segment["cent"])
    trgt = []
    othr = []

    for i in range(len(cent)):
        fframe = int(i / vfps * ffps)
        prev_idx = fframe + 1 - ac_feature_width

        trgt_window = segment["trgt"][max(prev_idx, 0) : fframe + 1].flatten()
        othr_window = segment["othr"][max(prev_idx, 0) : fframe + 1].flatten()

        if len(trgt_window) < ac_feature_width * ac_feature_size:
            dif = ac_feature_width - len(trgt_window) // ac_feature_size
            pad = np.zeros((dif * ac_feature_size), dtype=trgt_window.dtype)

            trgt_window = np.concatenate((pad, trgt_window), axis=0)
            othr_window = np.concatenate((pad, othr_window), axis=0)

        trgt.append(trgt_window)
        othr.append(othr_window)

    trgt = torch.tensor(np.stack(trgt))
    othr = torch.tensor(np.stack(othr))

    ans_cent = cent[1:].clone().detach().requires_grad_(True)
    cent = cent[:-1]

    ans_angl = angl[1:].clone().detach().requires_grad_(True)
    angl = angl[:-1]

    trgt = trgt[:-1].clone().detach().requires_grad_(True)
    othr = othr[:-1].clone().detach().requires_grad_(True)

    return ([angl, cent, trgt, othr], [ans_angl, ans_cent], vfps)


class Video:
    def __init__(self, video_path: str, codec: str = "mp4v") -> None:
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*codec)

        self.path = video_path
        self.name = os.path.basename(video_path)
        self.codec = codec

        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.writer = None

        self.step = 1

        self.current_idx = 0

        self.length = None
        self.__len__()

    def __str__(self) -> str:
        return f"all frame : {self.cap_frames}, fps : {round(self.fps, 2)}, time : {round(self.cap_frames/self.fps, 2)}"

    def __getitem__(self, idx):
        pos = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        return ret, frame

    def __len__(self) -> int:
        if self.length is None:
            self.length = math.ceil(self.cap_frames / self.step)
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx == self.length:
            self.reset()
            raise StopIteration

        frame = self.cap.read()[1]
        self.current_idx += 1
        for _ in range(self.step - 1):
            self.cap.read()
        return frame

    def reset(self):
        self.current_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def info(self):
        return [self.fourcc, self.cap_width, self.cap_height, self.fps, self.cap_frames]

    def read(self):
        return self.cap.read()

    def set_out_path(self, path: str):
        self.writer = cv2.VideoWriter(
            path, self.fourcc, self.fps, (self.cap_width, self.cap_height)
        )

    def write(self, frame):
        self.writer.write(frame)

    def set_step(self, step):
        self.step = step

    def close_writer(self):
        self.writer.release()


def visualize_result(
    video: Video,
    result: Tuple[torch.Tensor, torch.Tensor],
    landmarks: np.ndarray,
    path: str,
    tqdm_visual: bool,
):
    def frame_writer(frame: np.ndarray, param: Video):
        param.write(frame)

    video.set_out_path(path)

    if tqdm_visual:
        progress_iterator = zip(tqdm(result[0], desc="  visualize-all "), result[1])
    else:
        progress_iterator = zip(*result)

    for (angl, cent) in progress_iterator:
        frame = next(video)
        frame[:, :, :] = 0
        angl = np.array(angl.to(dtype=torch.float64))
        cent = np.array(cent.to(dtype=torch.float64))

        R = rotation_matrix(*angl, "xyz")

        _landmarks = np.dot(R, landmarks.copy().T).T + cent

        for pt in _landmarks:
            cv2.drawMarker(
                frame, (int(pt[0]), int(pt[1])), (254, 254, 254), cv2.MARKER_STAR, 2
            )

        frame_writer(frame, video)

    video.close_writer()
