from pathlib import Path

import chameleon as cl
import lightning as L
from lightning.pytorch.strategies import (DDPStrategy, DeepSpeedStrategy,
                                          FSDPStrategy, SingleDeviceStrategy)


class Trainer(L.Trainer):
    def save_checkpoint(self, filepath, weights_only) -> None:
        fpath = Path(filepath)
        super().save_checkpoint(filepath, weights_only)

        onnx_fpath = fpath.parent / fpath.name.replace('.ckpt', '.onnx')
        self.lightning_module.to_onnx(onnx_fpath)


STRATEGY = cl.Registry('train_strategy')

STRATEGY.register_module(module=DDPStrategy)
STRATEGY.register_module(module=DeepSpeedStrategy)
STRATEGY.register_module(module=FSDPStrategy)
STRATEGY.register_module(module=SingleDeviceStrategy)
