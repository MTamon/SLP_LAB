"""Simple-Model, it is the first step in this research"""

from typing import List, Tuple

import torch
from torch import nn

from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor


class SimpleModel(nn.Module):
    def __init__(
        self,
        lstm_dim: int,
        ac_linear_dim: int,
        lstm_input_dim: int,
        lstm_output_dim: int,
        pos_feature_size: int,
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
        self.pos_feature_size = pos_feature_size
        self.ac_feature_size = ac_feature_size
        self.ac_feature_width = ac_feature_width
        self.num_layer = num_layer

        self.device = torch.device("cpu") if device is None else device

        self.input_ac_size = ac_feature_size * ac_feature_width

        net_t = self.transfer_learning(vgg16(pretrained=True))
        net_o = self.transfer_learning(vgg16(pretrained=True))

        img_size = 3 * 244 * 244

        self.link_vgg_trgt = nn.Linear(self.input_ac_size, img_size).to(self.device)
        self.link_vgg_othr = nn.Linear(self.input_ac_size, img_size).to(self.device)
        self.extractor_trgt = create_feature_extractor(net_t, {"avgpool": "feature"})
        self.extractor_othr = create_feature_extractor(net_o, {"avgpool": "feature"})

        self.link_fc_trgt = nn.Linear(25088, ac_linear_dim).to(self.device)
        self.link_fc_othr = nn.Linear(25088, ac_linear_dim).to(self.device)
        self.cent_fc = nn.Linear(3, pos_feature_size).to(self.device)
        self.angl_fc = nn.Linear(3, pos_feature_size).to(self.device)

        self.cat_dim = pos_feature_size * 2 + ac_linear_dim * 2

        self.forward_lstm = nn.Linear(self.cat_dim, self.lstm_input_dim).to(self.device)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_dim,
            batch_first=True,
            num_layers=num_layer,
            bidirectional=False,
        ).to(self.device)

        self.backward_lstm = nn.Linear(lstm_dim, self.lstm_output_dim).to(self.device)

        self.angl_linear1 = nn.Linear(lstm_output_dim, lstm_output_dim).to(self.device)
        self.cent_linear1 = nn.Linear(lstm_output_dim, lstm_output_dim).to(self.device)

        self.angl_linear2 = nn.Linear(lstm_output_dim, 3).to(self.device)
        self.cent_linear2 = nn.Linear(lstm_output_dim, 3).to(self.device)

    def transfer_learning(self, net: nn.Module):
        for param in net.parameters():
            param.requires_grad = False

        return net.to(device=self.device)

    def input2vgg(self, input_t: torch.Tensor):
        """(batch, seq, ac_f_w*ac_f_s) -> (batch * seq, 1 * ac_f_w * ac_f_s)"""

        _shape = input_t.shape
        batch = _shape[0]
        seq = _shape[1]

        input_t = input_t.contiguous()
        input_t = input_t.view(size=(_shape[0] * _shape[1], _shape[2]))
        return input_t, (batch, seq)

    def vgg2lstm(self, input_t: torch.Tensor, b, s):
        """(batch, seq, ac_f_w*ac_f_s) -> (batch * seq, 1 * ac_f_w * ac_f_s)"""

        input_t = input_t.contiguous()
        input_t = input_t.view(size=(b, s, -1))
        return input_t

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

        _ac_ft_trgt, (b, s) = self.input2vgg(ac_ft_trgt)
        _ac_ft_othr, (b, s) = self.input2vgg(ac_ft_othr)

        _ac_ft_trgt = self.link_vgg_trgt(_ac_ft_trgt)
        _ac_ft_othr = self.link_vgg_othr(_ac_ft_othr)

        _ac_ft_trgt = self.extractor_trgt(_ac_ft_trgt)["feature"]
        _ac_ft_othr = self.extractor_othr(_ac_ft_othr)["feature"]

        _ac_ft_trgt = self.vgg2lstm(_ac_ft_trgt, b, s)
        _ac_ft_othr = self.vgg2lstm(_ac_ft_othr, b, s)

        _ac_ft_trgt = self.link_fc_trgt(_ac_ft_trgt)
        _ac_ft_othr = self.link_fc_othr(_ac_ft_othr)

        angl = self.angl_fc(angl)
        cent = self.angl_fc(cent)

        input_lstm = torch.cat([angl, cent, _ac_ft_trgt, _ac_ft_othr], axis=-1)

        input_lstm = self.forward_lstm(input_lstm)

        hn, (h, c) = self.lstm(input_lstm, (h_0, c_0))

        hn = self.backward_lstm(hn)

        _angl = self.angl_linear1(hn)
        _cent = self.cent_linear1(hn)

        _pred_angl = self.angl_linear2(_angl)
        _pred_cent = self.cent_linear2(_cent)

        return ((_pred_angl, _pred_cent), (h, c))
