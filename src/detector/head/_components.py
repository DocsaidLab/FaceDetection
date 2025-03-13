from typing import List, Union

import torch
import torch.nn as nn
from chameleon import BLOCKS, PowerModule


class Scale(PowerModule):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class ComponentHeadFeature(nn.Module):
    def __init__(
        self,
        in_channel: int = 64,
        hid_channel: int = 64,
        n_stack: int = 1,
        norm: dict = {"name": "BatchNorm2d", "num_features": 64},
        act: dict = {"name": "ReLU", "inplace": True},
        use_dwconv: bool = False,
    ):
        super().__init__()
        convs = [
            BLOCKS.build(
                {
                    "name": "SeparableConv2dBlock" if use_dwconv else "Conv2dBlock",
                    "in_channels": hid_channel if i else in_channel,
                    "out_channels": hid_channel,
                    "kernel": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": False,
                    "norm": norm,
                    "act": act,
                }
            )
            for i in range(n_stack)
        ]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class ComponentHead(nn.Module):
    def __init__(
        self,
        in_channel: int = 512,
        n_priors: int = 3,
        n_outputs: Union[int, List[int]] = 10,
        use_scale: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, n_priors * n_outputs, 3, 1, 1, bias=True)
        self.scale = Scale() if use_scale else None

    def forward(self, x):
        out = self.conv(x)
        out = self.scale(out) if self.scale is not None else out
        return out
