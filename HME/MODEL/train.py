"""This program is for training Simple-Model"""

import torch
from torch import nn

from dataloader import Dataloader


class Trainer:
    def __init__(self, net: nn.Module, dataloader: Dataloader) -> None:
        self.net = net
        self.dataloader = dataloader

    def __call__(self):
        pass
