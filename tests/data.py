import random

import capybara as cb
import numpy as np
from fire import Fire

from src.data import DATASETS, plot_data
from src.utils import setup_seed

setup_seed(0)


def main():
    train_dataset = DATASETS.build(
        dict(
            name='DatasetFromJson',
            txt_path='/data/data1/face_detection/processed/widerface_train/all.txt',
            transforms=[
                dict(
                    name='RandomSquareCrop',
                    crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                    pad_val=(128, 128, 128),
                ),
                dict(
                    name='Resize',
                    h=640,
                    w=640,
                ),
                dict(
                    name='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18,
                    gray_prob=0,
                ),
                dict(
                    name='Normalize',
                    mean=(127.5, 127.5, 127.5),
                    std=(128, 128, 128),
                    to_rgb=True
                ),
            ],
        )
    )
    for data in train_dataset:
        plotted = plot_data(data)
        cb.imwrite(plotted, 'test.png')
        breakpoint()


Fire(main)
