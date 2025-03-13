import capybara as cb
from fire import Fire
from pytorch_lightning import Trainer

from lightning_module import LightningModule


def main(
    ckpt_path: str,
    cfg_path: str = None,
    use_single_gpu: bool = False,
    debug: bool = False,
):
    assert cfg_path is not None or ckpt_path is not None

    cfg_path = cb.Path(ckpt_path).parent.parent / "cfg.yaml" if cfg_path is None else cfg_path
    cfg = cb.PowerDict.load_yaml(cfg_path)

    if use_single_gpu:
        cfg.trainer.strategy = None
        cfg.trainer.devices = [0]

    lm = LightningModule(cfg.lightning_module, ckpt_path=ckpt_path)
    trainer = Trainer(
        fast_dev_run=debug,
        **cfg["trainer"],
    )
    trainer.validate(lm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    Fire(main)
