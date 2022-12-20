from model.ohta.modules.ContextNet.contextnet.convolution import ConvBlock

from typing import List
import torch
import torch.nn as nn


class AcosticCat(nn.Module):
    def __init__(self) -> None:
        super(AcosticCat, self).__init__()

    def forward(self, acf1, acf2):
        """Concatenate acostic feature 1, 2.

        Args:
            acf1 (torch.FloatTensor): ``(batch, seq, dim1)``
            acf2 (torch.FloatTensor): ``(batch, seq, dim2)``
        """

        return torch.cat((acf1, acf2), dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        acostic_dim: int = 80,
        num_layers: int = 5,
        kernel_size: int = 5,
        num_channels: int = 128,
        cnet_out_dim: int = 320,
        output_dim: int = 512,
        physic_frame_width: int = 10,
        acostic_frame_width: int = 69,
        use_power: bool = True,
        use_delta: bool = True,
        use_person: tuple = (True, True),
    ) -> None:
        super(Encoder, self).__init__()

        self.acostic_dim = acostic_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.cnet_out_dim = cnet_out_dim
        self.acostic_frame_width = acostic_frame_width
        self.use_power = use_power
        self.use_delta = use_delta
        self.use_person = use_person

        afe_num = sum(use_person) * 2 if use_delta else 1
        physic_out = output_dim // 2
        self.afe_out = output_dim // afe_num // 2

        physic_frame_size = 12 if use_delta else 6
        physic_size = physic_frame_size * physic_frame_width

        self.AFEs = self._get_afe_dict()

        self.PFE = PhysicSet(physic_size, physic_out)

        modules_out_size = physic_out + self.afe_out * afe_num

        self.final = nn.Linear(modules_out_size, output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def _get_afe_dict(self):
        afe_inputs = (
            self.acostic_dim,
            self.num_layers,
            self.kernel_size,
            self.num_channels,
            self.cnet_out_dim,
            self.afe_out,
            self.acostic_frame_width,
            self.use_power,
        )

        AFEs = nn.ModuleDict(
            {
                "AFE1": None,
                "AFE2": None,
                "dAFE1": None,
                "dAFE2": None,
            }
        )

        if self.use_person[0]:
            AFEs["AFE1"] = AcosticSet(*afe_inputs)
        if self.use_person[1]:
            AFEs["AFE2"] = AcosticSet(*afe_inputs)
        if self.use_person[0] and self.use_delta:
            AFEs["dAFE1"] = AcosticSet(*afe_inputs)
        if self.use_person[1] and self.use_delta:
            AFEs["dAFE2"] = AcosticSet(*afe_inputs)

        return AFEs

    def _fbank_process(
        self,
        fbank1=(None, None),
        fbank2=(None, None),
        log_power1=(None, None),
        log_power2=(None, None),
    ):
        if self.use_person[0]:
            out_afe = self.AFEs["AFE1"](fbank1[0], log_power1[0])
        if self.use_person[1]:
            _out_afe = self.AFEs["AFE2"](fbank2[0], log_power2[0])
            out_afe = torch.cat((out_afe, _out_afe), dim=-1)
        if self.use_person[0] and self.use_delta:
            _out_afe = self.AFEs["dAFE1"](fbank1[1], log_power1[1])
            out_afe = torch.cat((out_afe, _out_afe), dim=-1)
        if self.use_person[1] and self.use_delta:
            _out_afe = self.AFEs["dAFE2"](fbank2[1], log_power2[1])
            out_afe = torch.cat((out_afe, _out_afe), dim=-1)

        return out_afe

    def forward(self, input_tensor: List[List[torch.Tensor]]):
        """
        Args:
            input_tensor (List[torch.Tensor]): ``[angle, centroid, fbank1, fbank2, log_power1, log_power2]``
        """
        angle = input_tensor[0]
        centroid = input_tensor[1]
        acostic_input = input_tensor[2:]

        if self.use_delta:
            pass

        batch_size = angle[0].shape[0]

        physic_tensor = torch.cat((*angle, *centroid), dim=-1)
        physic_tensor = physic_tensor.view(size=[batch_size, -1])
        physic_output = self.PFE(physic_tensor)

        acostic_output = self._fbank_process(*acostic_input)

        feature = torch.cat((physic_output, acostic_output), dim=-1)

        output = self.final(feature)
        output = self.norm(output)

        return output


class AcosticSet(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        num_layers: int = 5,
        kernel_size: int = 5,
        num_channels: int = 128,
        cnet_out_dim: int = 320,
        output_dim: int = 64,
        acostic_frame_width: int = 69,
        use_power: bool = True,
    ) -> None:
        super(AcosticSet, self).__init__()

        self.input_dim = input_dim
        self.acostic_frame_width = acostic_frame_width
        self.use_power = use_power
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.acostic_enc = ConvBlock.make_conv_blocks(
            input_dim, num_layers, kernel_size, num_channels, cnet_out_dim
        )
        if use_power:
            power_out = cnet_out_dim // 4
            power_channels = num_channels // 4
            self.power_enc = ConvBlock.make_conv_blocks(
                1, num_layers, kernel_size, power_channels, power_out
            )
            self.acf_cat = AcosticCat()

        self.output_seq = self.get_output_seq()
        self.dense_input = (cnet_out_dim + power_out) if use_power else cnet_out_dim

        self.dence1 = nn.Linear(self.dense_input * self.output_seq, self.dense_input)
        self.dense2 = nn.Linear(self.dense_input, output_dim)

    def get_output_seq(self):
        input_tensor = torch.zeros(
            (1, self.input_dim, self.acostic_frame_width), dtype=torch.float32
        )
        length = torch.tensor([self.acostic_frame_width], dtype=torch.int32)
        output_tensor = input_tensor.to(device=self.device)
        output_length = length.to(device=self.device)

        for block in self.acostic_enc:
            output_tensor, output_length = block(output_tensor, output_length)
        return int(output_length[0])

    def forward(self, fbank: torch.FloatTensor, log_power: torch.FloatTensor = None):
        """
        Args:
            fbank (torch.FloatTensor): ``(batch, seq)``
            log_power (torch.FloatTensor, optional): ``(batch, seq, dim)``. Defaults to None.

        Returns:
            torch.FloatTensor: ``(batch, output_dim)``
        """
        output = fbank.transpose(1, 2)
        length = torch.tensor(fbank.shape[1]).repeat(fbank.shape[0])
        output_length = length.to(device=self.device)

        if self.use_power:
            iterator = zip(self.acostic_enc, self.power_enc)
            power_length = length
            power_output = log_power.unsqueeze(1)
        else:
            iterator = self.acostic_enc

        for block in iterator:
            if self.use_power:
                pblock = block[1]
                block = block[0]
                power_output, power_length = pblock(power_output, power_length)

            output, output_length = block(output, output_length)

        if self.use_power:
            output = self.acf_cat(output.transpose(1, 2), power_output.transpose(1, 2))
        else:
            output = output.transpose(1, 2)

        output = output.view(size=[fbank.shape[0], -1])
        output = self.dence1(output)
        output = self.dense2(output)

        return output


class PhysicSet(nn.Module):
    def __init__(self, input_dim: int = 120, output_dim: int = 256) -> None:
        super(PhysicSet, self).__init__()

        self.hidden_size1 = input_dim * 3
        self.hidden_size2 = self.hidden_size1 // 2
        self.hidden_size3 = self.hidden_size2 // 4
        self.hidden_size4 = 32

        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, self.hidden_size3),
            nn.ReLU(),
            nn.Linear(self.hidden_size3, self.hidden_size4),
            nn.ReLU(),
            nn.Linear(self.hidden_size4, self.hidden_size4),
            nn.ReLU(),
            nn.Linear(self.hidden_size4, output_dim),
        )

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_tensor (torch.Tensor): ``[batch, (centorid + Δcentorid + angle + Δangle) * flame]``

        Returns:
            torch.FloatTensor: ``[batch, output_dim]``
        """

        output = self.net(input_tensor)

        return output
