import capybara as cb
from fire import Fire

from src.lightning_module import LightningModule


def main(
    cfg_path: str,
    ckpt_path: str = None,
    onnx_path: str = None,
):
    cfg_path = cb.Path(cfg_path)
    cfg = cb.load_yaml(cfg_path)
    lm = LightningModule(cfg["lightning_module"], ckpt_path=ckpt_path)
    if ckpt_path is None and onnx_path is None:
        onnx_path = "tmp.onnx"
    elif ckpt_path is not None and onnx_path is None:
        onnx_path = ckpt_path.replace(".ckpt", ".onnx")
    lm.to_onnx(onnx_path, verbose=True)


if __name__ == "__main__":
    Fire(main)
