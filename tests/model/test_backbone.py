from chameleon import BACKBONES, calculate_flops
from src.detector.backbone import *


def test_scrfd_500m():
    model = BACKBONES.build({'name': 'scrfd_mbv1_500m'})
    assert isinstance(model, MobileNetV1)
    flops, macs, params = calculate_flops(model, (1, 3, 640, 640))
    assert flops == '1.03 GFLOPS'
    assert macs == '497.66 MMACs'
    assert params == '536.23 K'


def test_scrfd_10g():
    model = BACKBONES.build({'name': 'scrfd_resv1d_10g'})
    assert isinstance(model, ResNet)
    flops, macs, params = calculate_flops(model, (1, 3, 640, 640))
    assert flops == '23 GFLOPS'
    assert macs == '11.46 GMACs'
    assert params == '3.48 M'
