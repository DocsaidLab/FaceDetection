import random

import capybara as cb
import numpy as np
from fire import Fire

from src.data import DATASETS, plot_data
from src.utils import setup_seed

setup_seed(0)


def main():
    train_dataset1 = DATASETS.build(
        dict(
            name='DatasetFromJson',
            txt_path='/data/data1/face_detection/processed/widerface_train/all.txt',
            transforms=[
                dict(
                    name='RandomSquareCropSCRFD',
                    crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                    pad_val=(128, 128, 128),
                ),
                dict(
                    name='Resize',
                    h=640,
                    w=640,
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
    train_dataset2 = DATASETS.build(
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
                    name='Normalize',
                    mean=(127.5, 127.5, 127.5),
                    std=(128, 128, 128),
                    to_rgb=True
                ),
            ],
        )
    )
    for data1, data2 in zip(train_dataset1, train_dataset2):
        plotted1 = plot_data(data1)
        plotted2 = plot_data(data2)
        cb.imwrite(plotted1, 'test1.png')
        cb.imwrite(plotted2, 'test2.png')
        breakpoint()


Fire(main)
