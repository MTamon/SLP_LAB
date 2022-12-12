"""Simple-Model, it is the first step in this research"""

from typing import List, Tuple

import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(
        self,
        lstm_dim: int,
        ac_linear_dim: int,
        lstm_input_dim: int,
        lstm_output_dim: int,
        ac_feature_size: int,
        ac_feature_width: int,
        num_layer: int,
        *args,
        device: torch.device = None,
        **kwargs
    ):
        super(SimpleModel, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.lstm_dim = lstm_dim
        self.ac_linear_dim = ac_linear_dim
        self.lstm_input_dim = lstm_input_dim
        self.lstm_output_dim = lstm_output_dim
        self.ac_feature_size = ac_feature_size
        self.ac_feature_width = ac_feature_width
        self.num_layer = num_layer

        self.device = "cpu" if device is None else device

        self.input_ac_size = ac_feature_size * ac_feature_width

        self.input_ac_feature1a = nn.Linear(self.input_ac_size, ac_feature_size)
        self.input_ac_feature2a = nn.Linear(self.input_ac_size, ac_feature_size)
        self.input_ac_feature1b = nn.Linear(ac_feature_size, ac_linear_dim)
        self.input_ac_feature2b = nn.Linear(ac_feature_size, ac_linear_dim)

        self.cat_input_dim = 3 + 3 + ac_linear_dim + ac_linear_dim

        self.forward_lstm = nn.Linear(self.cat_input_dim, self.lstm_input_dim)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_dim,
            batch_first=True,
            num_layers=num_layer,
            bidirectional=False,
        )

        self.backward_lstm = nn.Linear(lstm_dim, self.lstm_output_dim)

        self.angl_linear1 = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.cent_linear1 = nn.Linear(lstm_output_dim, lstm_output_dim)

        self.angl_linear2 = nn.Linear(lstm_output_dim, 3)
        self.cent_linear2 = nn.Linear(lstm_output_dim, 3)

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

        angl = input_tensor[0]
        cent = input_tensor[1]
        ac_ft_trgt = input_tensor[2]
        ac_ft_othr = input_tensor[3]

        _ac_ft_trgt = self.input_ac_feature1a(ac_ft_trgt)
        _ac_ft_othr = self.input_ac_feature2a(ac_ft_othr)

        _ac_ft_trgt = self.input_ac_feature1b(_ac_ft_trgt)
        _ac_ft_othr = self.input_ac_feature2b(_ac_ft_othr)

        input_lstm = torch.cat([angl, cent, _ac_ft_trgt, _ac_ft_othr], axis=-1)

        input_lstm = self.forward_lstm(input_lstm)

        hn, (h, c) = self.lstm(input_lstm, (h_0, c_0))

        hn = self.backward_lstm(hn)

        _angl = self.angl_linear1(hn)
        _cent = self.cent_linear1(hn)

        _pred_angl = self.angl_linear2(_angl)
        _pred_cent = self.cent_linear2(_cent)

        return ((_pred_angl, _pred_cent), (h, c))
