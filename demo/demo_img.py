import capybara as cb
from fire import Fire

from src.onnx.scrfd import SCRFD
from src.utils import draw_results


def main(img_fpath: str, onnx_fpath: str):
    img = cb.imread(img_fpath)
    model = SCRFD(onnx_fpath, score_th=0.3, backend='cuda')
    proposals = model([img])[0]
    plotted = draw_results(
        img,
        boxes=cb.Boxes(proposals['boxes']),
        scores=proposals['scores'],
        kpts_list=cb.KeypointsList(proposals['lmk5pts']),
    )
    cb.imwrite(plotted, 'demo.png')


Fire(main)
