import capybara as cb
import numpy as np
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DATASETS, collate_fn, plot_batch_data


def main():
    dataset = DATASETS.build(
        dict(
            name="ConcatDatasets",
            datasets=[
                dict(
                    name="DatasetFromJson",
                    txt_path="/data/data1/face_detection/processed/widerface_train/all.txt",
                    transforms=[
                        dict(
                            name="ToFloat32",
                            keys=["image", "boxes", "lmk5pts"],
                        ),
                        dict(
                            name="RandomSquareCrop",
                            crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                            pad_val=128,
                        ),
                    ],
                ),
            ],
            transforms=[
                dict(
                    name="Resize",
                    h=640,
                    w=640,
                    enable_random_interpolation=True,
                ),
                dict(
                    name="PhotoMetricDistortion",
                    gray_prob=0.1,
                ),
                dict(
                    name="RandomHFlip",
                    prob=0.5,
                ),
                dict(
                    name="Normalize",
                    mean=127.5,
                    std=128,
                ),
            ],
        )
    )
    # dataset = DATASETS.build(
    #     dict(
    #         name='DatasetFromJson',
    #         txt_path='data/public/widerface_train/valid.txt',
    #         transforms=[
    #             dict(
    #                 name='ResizePadIfNeed',
    #                 h=640,
    #                 w=640,
    #             ),
    #             dict(
    #                 name='Normalize',
    #                 mean=127.5,
    #                 std=128,
    #             ),
    #         ],
    #     )

    # )
    # sampler = SAMPLERS.build(
    #     {
    #         'name': 'MultiScaleBatchSampler',
    #         'dataset': dataset,
    #         'size_candidates': [384, 512, 640, 768, 896],
    #         'batch_size': 16,
    #         'use_ddp': False,
    #         'shuffle': True,
    #         'drop_last': True,
    #     }
    # )
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        batch_size=16,
        collate_fn=collate_fn,
    )
    for batch in tqdm(dataloader, desc="Dataloader iterating...", unit_scale=16):
        plotteds = plot_batch_data(batch)
        plotted = [np.concatenate(x, axis=1) for x in cb.make_batch(plotteds, 4)]
        plotted = np.concatenate(plotted, axis=0)
        cb.imwrite(plotted, "test.png")
        # pass
        breakpoint()


Fire(main)
