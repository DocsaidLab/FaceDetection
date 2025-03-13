from datetime import datetime
from pathlib import Path

import capybara as cb
from fire import Fire
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from termcolor import cprint

from benchmark import nist, widerface
from src.callbacks import build_callback
from src.lightning_module import LightningModule
from src.utils import Trainer, setup_opencv, setup_torch

setup_opencv()
setup_torch()


def get_current_time_str():
    t = cb.now("datetime")
    new_t = datetime(t.year, t.month, t.day, t.hour, t.minute // 10 * 10, 0, 0)
    return cb.datetime2str(new_t, fmt="%Y%m%d%H%M")


def main(
    cfg_path: str,
    ckpt_path: str = None,
    resume_training: bool = False,
    profiler: bool = False,
    debug: bool = False,
):
    cfg = cb.load_yaml(cfg_path)

    if debug or profiler:
        cfg["trainer"]["strategy"] = "auto"
        cfg["trainer"]["devices"] = [0]
        cfg["trainer"]["max_epochs"] = 1
        cfg["trainer"]["limit_train_batches"] = 10
        cfg["lightning_module"]["dataloader"]["train"]["total_batch_size"] = 16
        cfg["lightning_module"]["dataloader"]["train"]["num_workers_per_node"] = 0
        cfg["lightning_module"]["dataloader"]["valid"]["total_batch_size"] = 16
        cfg["lightning_module"]["dataloader"]["valid"]["num_workers_per_node"] = 0
        cfg["lightning_module"]["widerface_evaluation"]["enable"] = False
        profiler = SimpleProfiler()

    # prepare lightning model
    # ---- naming experiment ---- #

    # logger
    log_dir = Path("results", cfg["phase"])
    log_dir.mkdir(exist_ok=True, parents=True)
    cfg_path = Path(cfg_path)

    current_time = get_current_time_str()
    logger = TensorBoardLogger(
        save_dir=log_dir,
        version=current_time,
        name=cfg_path.stem,
    )
    result_folder = Path(logger.log_dir)
    result_folder.mkdir(parents=True, exist_ok=True)
    print("log_dir:", result_folder)
    # log configs
    cb.dump_yaml(cfg, result_folder / "cfg.yaml")

    # ckpt
    ckpt_dir = result_folder / "ckpt"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    callbacks = [build_callback("ModelCheckpoint", dirpath=ckpt_dir, **cfg["checkpointer"])]

    # other callbacks
    callbacks += [build_callback(**callbacks) for callbacks in cfg["callbacks"]]
    trainer = Trainer(
        fast_dev_run=debug,
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        **cfg["trainer"],
    )

    lm = LightningModule(cfg["lightning_module"], ckpt_path=ckpt_path)
    if resume_training:
        cprint(f"Resume training from ckpt: {ckpt_path}", "yellow", attrs=["bold"])
        trainer.fit(lm, ckpt_path=ckpt_path)
    else:
        trainer.fit(lm)

    if trainer.is_global_zero:
        cprint("Finish training...", "green", attrs=["bold"])
        cprint("Widerface Evaluation...", "green", attrs=["bold"])
        onnx_fpath = ckpt_dir / "last.onnx"
        lm.to_onnx(onnx_fpath)

    trainer.strategy.barrier()
    del lm, trainer

    if _get_rank() == 0:
        widerface.main(onnx_fpath=onnx_fpath, **cfg["lightning_module"]["widerface_evaluation"])
        nist.main(onnx_fpath=onnx_fpath, **cfg["lightning_module"]["nist_evaluation"])


if __name__ == "__main__":
    Fire(main)
