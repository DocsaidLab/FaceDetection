from typing import Dict

import pandas as pd
from lightning.pytorch.loggers import Logger


class PandasLogger(Logger):
    def __init__(self, save_dir: str, name: str, version: str):
        super().__init__()
        self.save_dir = save_dir
        self.name = name
        self.version = version

    def log_metrics(self, metrics: Dict[str, float], step: int):
        pass

    def log_hyperparams(self, params: Dict[str, any]):
        pass

    def finalize(self):
        pass

    def close(self):
        pass
