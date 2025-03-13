from typing import List, Optional, Union

import torch.nn as nn
from chameleon import BLOCKS, PowerModule


class BasicBlock(PowerModule):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        norm: dict = {'name': 'BatchNorm2d'},
        use_dwconv: bool = False,
        downsample: Optional[dict] = None,
    ):
        super().__init__()
        if norm['name'] == 'GroupNorm':
            norm.update({'num_channels': out_channels})
        else:
            norm.update({'num_features': out_channels})

        self.block1 = BLOCKS.build(
            {
                'name': 'SeparableConv2dBlock' if use_dwconv else 'Conv2dBlock',
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel': 3,
                'stride': stride,
                'padding': dilation,
                'dilation': dilation,
                'bias': False,
                'norm': norm,
                'act': {'name': 'ReLU', 'inplace': True},
            }
        )
        self.block2 = BLOCKS.build(
            {
                'name': 'SeparableConv2dBlock' if use_dwconv else 'Conv2dBlock',
                'in_channels': out_channels,
                'out_channels': out_channels,
                'kernel': 3,
                'padding': 1,
                'bias': False,
                'norm': norm,
            }
        )
        self.downsample = downsample if downsample is not None else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.act(identity + out)
        return out


class Bottleneck(PowerModule):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        norm: dict = {'name': 'BatchNorm2d'},
        use_dwconv: bool = False,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        if norm['name'] == 'GroupNorm':
            norm.update({'num_channels': out_channels})
        else:
            norm.update({'num_features': out_channels})
        self.block1 = BLOCKS.build(
            {
                'name': 'SeparableConv2dBlock' if use_dwconv else 'Conv2dBlock',
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel': 1,
                'stride': 1,
                'padding': 0,
                'bias': False,
                'norm': norm,
                'act': {'name': 'ReLU', 'inplace': True},
            }
        )
        self.block2 = BLOCKS.build(
            {
                'name': 'SeparableConv2dBlock' if use_dwconv else 'Conv2dBlock',
                'in_channels': out_channels,
                'out_channels': out_channels,
                'kernel': 3,
                'stride': stride,
                'padding': dilation,
                'dilation': dilation,
                'bias': False,
                'norm': norm,
            }
        )
        if norm['name'] == 'GroupNorm':
            norm.update({'num_channels': out_channels * self.expansion})
        else:
            norm.update({'num_features': out_channels * self.expansion})
        self.block3 = BLOCKS.build(
            {
                'name': 'SeparableConv2dBlock' if use_dwconv else 'Conv2dBlock',
                'in_channels': out_channels,
                'out_channels': out_channels * self.expansion,
                'kernel': 1,
                'stride': 1,
                'padding': 0,
                'norm': norm,
            }
        )
        self.downsample = downsample if downsample is not None else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.act(identity + out)
        return out


class ResLayer(nn.Sequential):
    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        norm: dict = {'name': 'BatchNorm2d'},
        avg_down: bool = False,
        use_dwconv: bool = False,
        downsample_first: bool = True,
        **kwargs
    ) -> None:
        self.block = block

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False
                    )
                )
            downsample.append(
                BLOCKS.build({
                    'name': 'SeparableConv2dBlock' if use_dwconv else 'Conv2dBlock',
                    'in_channels': in_channels,
                    'out_channels': out_channels * block.expansion,
                    'kernel': 1,
                    'stride': conv_stride,
                    'padding': 0,
                    'bias': False,
                    'norm': {'name': 'BatchNorm2d', 'num_features': out_channels * block.expansion},
                })
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    norm=norm,
                    use_dwconv=use_dwconv,
                    downsample=downsample,
                    **kwargs
                )
            )
            in_channels = out_channels * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        norm=norm,
                        use_dwconv=use_dwconv,
                        **kwargs
                    )
                )
        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        norm=norm,
                        use_dwconv=use_dwconv,
                        **kwargs
                    )
                )
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    norm=norm,
                    use_dwconv=use_dwconv,
                    downsample=downsample,
                    **kwargs
                )
            )
        super().__init__(*layers)


class ResNet(PowerModule):
    MetaParams = {
        '2.5g': {
            'block_cfg': {'block': BasicBlock, 'stage_blocks': [3, 5, 3, 2], 'stage_planes': [24, 48, 48, 80]},
            'base_channels': 24,
            'num_stages': 4,
            'norm': {'name': 'BatchNorm2d'},
        },
        '10g': {
            'block_cfg': {'block': BasicBlock, 'stage_blocks': [3, 4, 2, 3], 'stage_planes': [56, 88, 88, 224]},
            'base_channels': 56,
            'num_stages': 4,
            'norm': {'name': 'BatchNorm2d'},
        },
        '34g': {
            'block_cfg': {'block': Bottleneck, 'stage_blocks': [17, 16, 2, 8], 'stage_planes': [56, 56, 144, 184]},
            'base_channels': 56,
            'num_stages': 4,
            'norm': {'name': 'BatchNorm2d'},
        }
    }

    def __init__(
        self,
        block_cfg: dict,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: tuple = (1, 2, 2, 2),
        dilations: tuple = (1, 1, 1, 1),
        out_indices: tuple = (0, 1, 2, 3),
        deep_stem: bool = False,
        avg_down: bool = False,
        no_pool33: bool = False,
        use_dwconv: bool = False,
        norm: dict = {'name': 'BatchNorm2d'},
    ):
        super().__init__()

        block = block_cfg['block']
        stage_blocks = block_cfg['stage_blocks']

        if num_stages < 1 and num_stages > 4:
            raise ValueError('num_stages should be in [1, 4]')

        if len(strides) != num_stages or len(dilations) != num_stages:
            raise ValueError('strides and dilations should have the same length as num_stages')

        if len(stage_blocks) < num_stages:
            raise ValueError(f'stage_blocks should have at least {num_stages} elements')

        stage_blocks = stage_blocks[:num_stages]
        stage_planes = block_cfg['stage_planes']

        self.stem = self._make_stem(
            in_channels=in_channels,
            stem_channels=base_channels,
            deep=deep_stem,
        )
        if no_pool33:
            self.stem_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.stem_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = nn.ModuleDict()
        self.feature_channels = []
        in_channels = base_channels
        for i, (num_blocks, planes, stride, dilation) in enumerate(zip(stage_blocks, stage_planes, strides, dilations)):
            norm['num_features'] = planes
            self.res_layers[f'layer{i + 1}'] = ResLayer(
                block=block,
                in_channels=in_channels,
                out_channels=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                norm=norm,
                avg_down=avg_down,
                use_dwconv=use_dwconv,
            )
            in_channels = planes * block.expansion

        self.feature_channels = [planes * block.expansion for planes in stage_planes]
        self.out_indices = out_indices if out_indices is not None else list(range(len(self.res_layers) + 1))
        self.initialize_weights_()

    def _make_stem(self, in_channels, stem_channels, deep: bool = False) -> nn.Module:
        if deep:
            stem = nn.Sequential(
                BLOCKS.build(
                    {
                        'name': 'Conv2dBlock',
                        'in_channels': in_channels,
                        'out_channels': stem_channels // 2,
                        'kernel': 3,
                        'stride': 2,
                        'padding': 1,
                        'bias': False,
                        'norm': {'name': 'BatchNorm2d', 'num_features': stem_channels // 2},
                        'act': {'name': 'ReLU', 'inplace': True},
                    }
                ),
                BLOCKS.build(
                    {
                        'name': 'Conv2dBlock',
                        'in_channels': stem_channels // 2,
                        'out_channels': stem_channels // 2,
                        'kernel': 3,
                        'stride': 1,
                        'padding': 1,
                        'bias': False,
                        'norm': {'name': 'BatchNorm2d', 'num_features': stem_channels // 2},
                        'act': {'name': 'ReLU', 'inplace': True},
                    }
                ),
                BLOCKS.build(
                    {
                        'name': 'Conv2dBlock',
                        'in_channels': stem_channels // 2,
                        'out_channels': stem_channels,
                        'kernel': 3,
                        'stride': 1,
                        'padding': 1,
                        'bias': False,
                        'norm': {'name': 'BatchNorm2d', 'num_features': stem_channels},
                        'act': {'name': 'ReLU', 'inplace': True},
                    }
                )
            )
        else:
            stem = BLOCKS.build(
                {
                    'name': 'Conv2dBlock',
                    'in_channels': in_channels,
                    'out_channels': stem_channels,
                    'kernel': 7,
                    'stride': 2,
                    'padding': 3,
                    'bias': False,
                    'norm': {'name': 'BatchNorm2d', 'num_features': stem_channels},
                    'act': {'name': 'ReLU', 'inplace': True},
                }
            )
        return stem

    def forward(self, x):
        out = self.stem(x)
        outs = [out] if 0 in self.out_indices else []
        out = self.stem_maxpool(out)
        max_ind = max(self.out_indices)
        for i in range(1, max_ind + 1):
            out = self.res_layers[f'layer{i}'](out)
            if i in self.out_indices:
                outs.append(out)
        return outs

    @ classmethod
    def build_resnetv1d(cls, name: str, in_channels: int = 3, out_indices: Optional[List[int]] = None) -> nn.Module:
        """
        ResNetV1d variant described in `Bag of Tricks
        <https://arxiv.org/pdf/1812.01187.pdf>`_.

        Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
        the input stem with three 3x3 convs. And in the downsampling block, a 2x2
        avg_pool with stride 2 is added before conv, whose stride is changed to 1.
        """
        return cls(
            in_channels=in_channels,
            deep_stem=True,
            avg_down=True,
            out_indices=out_indices,
            **cls.MetaParams[name.split('scrfd_resv1d_')[-1]],
        )

    @ classmethod
    def build_resnetv1e(cls, name: str, in_channels: int = 3, out_indices: Optional[List[int]] = None) -> nn.Module:
        """
        ResNetV1d variant described in `Bag of Tricks
        <https://arxiv.org/pdf/1812.01187.pdf>`_.

        Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
        the input stem with three 3x3 convs. And in the downsampling block, a 2x2
        avg_pool with stride 2 is added before conv, whose stride is changed to 1.
        """
        return cls(
            in_channels=in_channels,
            deep_stem=True,
            avg_down=True,
            no_pool33=True,
            out_indices=out_indices,
            **cls.MetaParams[name.split('scrfd_resv1e_')[-1]],
        )
