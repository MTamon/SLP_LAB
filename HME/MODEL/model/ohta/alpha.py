from typing import List
import torch
import torch.nn as nn

from util import Encoder


model_size_rate = {"small": 0.5, "mediam": 1.0, "large": 2.0}


def build_alpha(
    model_type: str = "mediam",
    use_power: bool = True,
    use_delta: bool = True,
    use_person: tuple = (True, True),
    device: torch.device = None,
):
    assert (
        model_type in model_size_rate
    ), f"invalid model type {model_type}. {tuple(model_size_rate.keys())}"

    if device is None:
        device = torch.device("cpu")

    size_rate = model_size_rate[model_type]

    acostic_dim = 80
    num_layers = 5
    kernel_size = 5
    out_kernel_size = 5
    num_channels = int(128 * size_rate)
    cnet_out_dim = int(320 * size_rate)
    encoder_dim = int(512 * size_rate)
    physic_frame_width = 10
    acostic_frame_width = 69

    alpha = Alpha(
        acostic_dim,
        num_layers,
        kernel_size,
        out_kernel_size,
        num_channels,
        cnet_out_dim,
        encoder_dim,
        physic_frame_width,
        acostic_frame_width,
        use_power,
        use_delta,
        use_person,
    )
    alpha.to(device=torch.device(device))

    return alpha


class Alpha(nn.Module):
    def __init__(
        self,
        acostic_dim: int = 80,
        num_layers: int = 5,
        kernel_size: int = 5,
        out_kernel_size: int = 5,
        num_channels: int = 128,
        cnet_out_dim: int = 320,
        encoder_dim: int = 512,
        physic_frame_width: int = 10,
        acostic_frame_width: int = 69,
        use_power: bool = True,
        use_delta: bool = True,
        use_person: tuple = (True, True),
    ) -> None:
        super(Alpha, self).__init__()

        self.encoder = Encoder(
            acostic_dim=acostic_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            num_channels=num_channels,
            cnet_out_dim=cnet_out_dim,
            output_dim=encoder_dim,
            physic_frame_width=physic_frame_width,
            acostic_frame_width=acostic_frame_width,
            use_power=use_power,
            use_delta=use_delta,
            use_person=use_person,
        )

        self.dense = nn.Linear(encoder_dim, encoder_dim)
        self.norm = nn.LayerNorm(encoder_dim)

        self.out_conv = OutConv(encoder_dim, num_channels, out_kernel_size)
        out_conv_dim = self.out_conv.outputs_dim

        self.ch_modules = nn.ModuleDict(
            {
                "centroid": ChannelDense(out_conv_dim),
                "angle": ChannelDense(out_conv_dim),
            }
        )

    def forward(self, input_tensor: List[List[torch.Tensor]]):
        """
        Args:
            input_tensor (List[torch.Tensor]): ``[angle, centroid fbank1, fbank2, log_power1, log_power2]``
        """
        encoder_output = self.encoder(input_tensor)
        feature = self.dense(encoder_output)
        feature = self.norm(feature)

        outputs = self.out_conv(feature)
        centroid, angle = outputs.transpose(0, 1)

        centroid = self.ch_modules["centroid"](centroid)
        angle = self.ch_modules["angle"](angle)

        return centroid, angle


class OutConv(nn.Module):
    def __init__(
        self, input_dim: int = 512, num_channels: int = 256, kernel_size: int = 3
    ) -> None:
        super(OutConv, self).__init__()
        assert num_channels % 16 == 0, "Dimension should be divisible by 16."

        self.input_dim = input_dim

        self.blocks = nn.ModuleList()

        self.blocks.append(
            nn.Sequential(
                ConvLayer(1, num_channels, kernel_size, 1),
                ConvLayer(num_channels, num_channels, kernel_size, 1),
                ConvLayer(num_channels, num_channels, kernel_size, 1),
                ConvLayer(num_channels, num_channels, kernel_size, 1),
                ConvLayer(num_channels, num_channels, kernel_size, 2),
            )
        )
        self.blocks.append(
            nn.Sequential(
                ConvLayer(num_channels, num_channels * 2, kernel_size, 1),
                ConvLayer(num_channels * 2, num_channels * 2, kernel_size, 1),
                ConvLayer(num_channels * 2, num_channels, kernel_size, 2),
            )
        )
        self.blocks.append(
            nn.Sequential(
                ConvLayer(num_channels, num_channels // 2, kernel_size, 1),
                ConvLayer(num_channels // 2, num_channels // 4, kernel_size, 1),
                ConvLayer(num_channels // 4, num_channels // 8, kernel_size, 1),
                ConvLayer(num_channels // 8, num_channels // 16, kernel_size, 1),
                ConvLayer(num_channels // 16, 2, kernel_size, 2),
            )
        )

        self.outputs_dim = self._get_out_length()

    def _get_out_length(self):
        e = torch.zeros((1, 1, self.input_dim))
        out = e

        for block in self.blocks:
            out = block(out)

        return out.shape[-1]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): ``(batch, input_dim)``

        Returns:
            torch.Tensor: ``(batch, 2, outputs_dim)``
        """
        outputs = input_tensor.unsqueeze(1)

        for block in self.blocks:
            outputs = block(outputs)

        return outputs


class ConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super(ConvLayer, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        self.swish = Swish()

    def forward(self, input_tensor: torch.Tensor):
        outputs = self.conv1(input_tensor)

        outputs = self.batch_norm(outputs)
        outputs = self.swish(outputs)

        return outputs


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()


class ChannelDense(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(ChannelDense, self).__init__()
        assert int(input_dim * 0.4) > 3

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Linear(3, 3),
        )

    def forward(self, input_tensor: torch.Tensor):
        return self.net(input_tensor)
