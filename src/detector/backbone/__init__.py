from functools import partial

from chameleon import BACKBONES

from .mobilev1 import MobileNetV1
from .resnet import ResNet

BACKBONES.register_module(
    name="scrfd_mbv1_500m",
    module=partial(MobileNetV1.build_mobilenetv1, name="500m"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_mbv1_1g",
    module=partial(MobileNetV1.build_mobilenetv1, name="1g"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_resv1d_2.5g",
    module=partial(ResNet.build_resnetv1d, name="2.5g"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_resv1d_10g",
    module=partial(ResNet.build_resnetv1d, name="10g"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_resv1d_34g",
    module=partial(ResNet.build_resnetv1d, name="34g"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_resv1e_2.5g",
    module=partial(ResNet.build_resnetv1e, name="2.5g"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_resv1e_10g",
    module=partial(ResNet.build_resnetv1e, name="10g"),
    is_model_builder=True,
)
BACKBONES.register_module(
    name="scrfd_resv1e_34g",
    module=partial(ResNet.build_resnetv1e, name="34g"),
    is_model_builder=True,
)
