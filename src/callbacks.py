import lightning as L
import torch
from lightning.pytorch.callbacks import *  # noqa: F403
from lightning.pytorch.callbacks import ModelCheckpoint as PLModelCheckpoint
from torchvision.ops.boxes import box_iou


def get_obj_pairs(gt_boxes, pred_boxes, iou_th=0.9):
    ious = box_iou(gt_boxes, pred_boxes)
    inds = []
    if ious.shape[-1]:
        ious = torch.where(ious >= iou_th, ious, 0)
        inds = torch.argmax(ious, dim=1)
        inds = torch.where(ious.mean(dim=1) != 0, inds, -1).tolist()
    return inds


class ModelCheckpoint(PLModelCheckpoint):
    def on_exception(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        exception: BaseException,
    ) -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_last_checkpoint(trainer, monitor_candidates)


def build_callback(name, **kwargs):
    callback = globals().get(name, None)
    if callback is None:
        raise ValueError(f"Callback = {name} is not supported.")
    return callback(**kwargs)
