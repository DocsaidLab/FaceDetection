from typing import List, Optional

import torch.nn as nn
from chameleon import BLOCKS, PowerModule


class MobileNetV1(PowerModule):

    MetaParams = {
        '500m': {
            'stage_nplanes': [16, 16, 40, 72, 152, 288],
            'stage_nblocks': [2, 3, 2, 6],
        },
        '1g': {
            'stage_nplanes': [32, 48, 48, 160, 216, 312],
            'stage_nblocks': [3, 2, 1, 5],
        },
    }

    def __init__(
        self,
        in_channels: int = 3,
        stage_nplanes: List[int] = [16, 16, 40, 72, 152, 288],
        stage_nblocks: List[int] = [2, 3, 2, 6],
        out_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.stem = BLOCKS.build(
            {
                'name': 'Conv2dBlock',
                'in_channels': in_channels,
                'out_channels': stage_nplanes[0],
                'kernel': 3,
                'stride': 2,
                'padding': 1,
                'bias': False,
                'norm': {'name': 'BatchNorm2d', 'num_features': stage_nplanes[0]},
                'act': {'name': 'ReLU', 'inplace': True},
            }
        )
        self.stage0 = BLOCKS.build(
            {
                'name': 'SeparableConv2dBlock',
                'in_channels': stage_nplanes[0],
                'out_channels': stage_nplanes[1],
                'kernel': 3,
                'stride': 1,
                'padding': 1,
                'bias': False,
                'inner_norm': {'name': 'BatchNorm2d', 'num_features': stage_nplanes[0]},
                'inner_act': {'name': 'ReLU', 'inplace': True},
                'norm': {'name': 'BatchNorm2d', 'num_features': stage_nplanes[1]},
                'act': {'name': 'ReLU', 'inplace': True},
            }
        )

        self.channels = [stage_nplanes[1]]
        for i, nblocks in enumerate(stage_nblocks):
            in_channels = stage_nplanes[i + 1]
            out_channels = stage_nplanes[i + 2]
            stage = [
                BLOCKS.build(
                    {
                        'name': 'SeparableConv2dBlock',
                        'in_channels': out_channels if j else in_channels,
                        'out_channels': out_channels,
                        'kernel': 3,
                        'stride': 1 if j else 2,
                        'padding': 1,
                        'bias': False,
                        'inner_norm': {'name': 'BatchNorm2d', 'num_features': out_channels if j else in_channels},
                        'inner_act': {'name': 'ReLU', 'inplace': True},
                        'norm': {'name': 'BatchNorm2d', 'num_features': out_channels},
                        'act': {'name': 'ReLU', 'inplace': True},
                    }
                )
                for j in range(nblocks)
            ]
            stage = nn.Sequential(*stage)
            self.add_module(f'stage{i+1}', stage)
            self.channels.append(out_channels)

        self.out_indices = out_indices if out_indices is not None else list(range(len(stage_nblocks) + 1))
        self.initialize_weights_()

    def forward(self, xs):
        out = self.stem(xs)
        outs = []
        max_ind = max(self.out_indices)
        for i in range(max_ind + 1):
            out = getattr(self, f'stage{i}')(out)
            if i in self.out_indices:
                outs.append(out)
        return outs

    @classmethod
    def build_mobilenetv1(
        cls,
        name: str,
        in_channels: int = 3,
        out_indices: Optional[List[int]] = None,
    ):
        return cls(
            in_channels=in_channels,
            out_indices=out_indices,
            **cls.MetaParams[name],
        )
