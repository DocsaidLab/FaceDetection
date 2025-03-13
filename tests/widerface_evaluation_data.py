import capybara as cb
from fire import Fire

from src.data import DATASETS, plot_data


def main():
    train_dataset = DATASETS.build(
        dict(
            name="WiderFaceEvaluation",
            data_folder="data/public/widerface_val/data",
            gt_folder="data/public/widerface_val/evaluation_mat",
            transforms=[
                dict(
                    name="ResizePadIfNeed",
                    h=640,
                    w=640,
                ),
                dict(
                    name="Normalize",
                    mean=127.5,
                    std=128,
                ),
            ],
        )
    )
    for data in train_dataset:
        plotted = plot_data(data)
        cb.imwrite(plotted)
        breakpoint()
        # input('Press Enter to continue...')


Fire(main)
