"""This program define ReluUsed which based on SimpleModel"""

from typing import List, Tuple

import torch
from torch import nn

from utils import override
from simple import SimpleModel


class ReluUsed(SimpleModel):
    def __init__(
        self,
        lstm_dim: int,
        ac_linear_dim: int,
        lstm_input_dim: int,
        lstm_output_dim: int,
        pos_feature_size: int,
        relu_dim: int,
        dropout_rate: float,
        ac_feature_size: int,
        ac_feature_width: int,
        num_layer: int,
        *args,
        device: torch.device = None,
        **kwargs
    ):
        super().__init__(
            lstm_dim,
            ac_linear_dim,
            lstm_input_dim,
            lstm_output_dim,
            pos_feature_size,
            ac_feature_size,
            ac_feature_width,
            num_layer,
            *args,
            device=device,
            **kwargs
        )
        self.relu_dim = relu_dim
        self.dropout_rate = dropout_rate

        self.mixer1 = nn.Linear(self.lstm_output_dim, self.lstm_output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.mixer2 = nn.Linear(self.lstm_output_dim, relu_dim)
        self.relu = nn.ReLU()
        self.mixer3 = nn.Linear(relu_dim, self.lstm_output_dim)

    @override(SimpleModel)
    def forward(
        self,
        input_tensor: List[torch.Tensor],
        h_0: torch.Tensor = None,
        c_0: torch.Tensor = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input_tensor (List[torch.Tensor]): [angle, centroid, acostic_ft(target), acostic_ft(others)]
            h_0 (torch.Tensor): For setting value for LSTM's h. Default is None.
            c_0 (torch.Tensor): For setting value for LSTM's c. Default is None.
        output:
            pred_angl (torch.Tensor): predicted angle value.
            pred_cent (torch.Tensor): predicted centroid value.
            c (torch.Tensor): LSTM output c.
            h (torch.Tensor): LSTM output h.\\
                -------\\
            form: => ((pred_angl, pred_cent), (h, c))
        """
        batch_size = input_tensor[0].shape[0]
        if c_0 is None:
            c_0 = torch.zeros(
                (self.num_layer, batch_size, self.lstm_dim), device=self.device
            )
        if h_0 is None:
            h_0 = torch.zeros(
                (self.num_layer, batch_size, self.lstm_dim), device=self.device
            )

        input_lstm = self.forward_lstm(input_tensor)

        hn, (h, c) = self.lstm(input_lstm, (h_0, c_0))

        back = self.backward_lstm(hn)

        _angl, _cent = self.output_cent_angl(back)

        _angl += input_tensor[0]
        _cent += input_tensor[1]

        # Memory leak anti cuda-out-of-memory
        del h_0, c_0, back, hn, input_lstm, batch_size
        torch.cuda.empty_cache()

        return ((_angl, _cent), (h, c))

    @override(SimpleModel)
    def backward_lstm(self, hn: torch.Tensor):
        """LSTM backword process"""

        hn = self.link_lstm_back(hn)
        mixed = self.mixer1(hn)
        drop = self.dropout(mixed)
        mixed = self.mixer2(drop)
        mixed = self.relu(mixed)
        mixed = self.mixer3(mixed)

        return mixed
