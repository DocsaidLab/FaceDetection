import random
from functools import partial

import capybara as cb
import lightning as L
import numpy as np
import torch
from lightning.pytorch.strategies import SingleDeviceStrategy
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from benchmark import nist, widerface

from .data import DATASETS, collate_fn
from .detector import Detector
from .optim import OPTIMIZERS
from .utils import draw_results, get_model_params, restore_from_ckpt, setup_seed


def worker_init_fn(worker_id, seed):
    seed = seed + worker_id
    setup_seed(seed)


class LightningModule(L.LightningModule):
    def __init__(self, cfg, ckpt_path=None):
        super().__init__()
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self._build_model()

    def _build_model(self):
        self.model = Detector(**self.cfg["model"]).train()
        if self.ckpt_path is not None:
            ckpt = torch.load(self.ckpt_path, "cpu")["state_dict"]
            restore_from_ckpt(self, ckpt)

        torch.compile(self.model)

        self.metric = MeanAveragePrecision(iou_thresholds=[0.5])

    def configure_optimizers(self):
        solvers = []

        for cfg in self.cfg["solvers"]:
            solver = {}
            model_params = get_model_params(self.model, cfg.get("excludes", []))
            cfg["optimizer"].update({"params": model_params})
            solver["optimizer"] = OPTIMIZERS.build(cfg["optimizer"])

            # setting steps for lr_scheduler
            if "lr_scheduler" in cfg:
                if cfg["lr_scheduler"]["name"] == "OneCycleLR":
                    cfg["lr_scheduler"]["total_steps"] = self.trainer.estimated_stepping_batches
                elif cfg["lr_scheduler"]["name"] == "MultiStepLRWarmUp":
                    step_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
                    cfg["lr_scheduler"]["warmup_milestone"] *= int(step_per_epoch)
                    cfg["lr_scheduler"]["milestones"] = [
                        int(x * step_per_epoch) for x in cfg["lr_scheduler"]["milestones"]
                    ]
                cfg["lr_scheduler"]["optimizer"] = solver["optimizer"]
                solver["lr_scheduler"] = {
                    "scheduler": OPTIMIZERS.build(cfg["lr_scheduler"]),
                    "interval": "step",
                    "monitor": None,
                    "frequency": -1,
                }
            solvers.append(solver)
        return solvers

    # Dataloader
    def _build_dataloader(self, mode):
        dataset = DATASETS.build(self.cfg["dataset"][mode])
        num_workers = self.cfg["dataloader"][mode]["num_workers_per_node"]
        drop_last = not isinstance(self.trainer.strategy, SingleDeviceStrategy) and mode == "train"
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            shuffle=True if mode == "train" else False,
            batch_size=self.cfg["dataloader"][mode]["batch_size_per_device"],
            drop_last=drop_last,
            worker_init_fn=partial(worker_init_fn, seed=num_workers * self.local_rank),
        )
        return dataloader

    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("valid")

    def forward(self, x):
        return self.model(x)

    # steps
    def _step(self, mode, batch, batch_idx):
        outputs = {}
        losses = self.model.forward_train(batch)
        loss = losses.pop("loss")
        losses.pop("nt")

        if mode == "train":
            self.log(
                "loss",
                loss,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
                on_epoch=False,
            )
            self.log(
                "total_loss",
                loss,
                prog_bar=True,
                on_step=False,
                logger=True,
                sync_dist=True,
                on_epoch=True,
            )
            self.log_dict(
                losses,
                prog_bar=True,
                on_step=False,
                logger=True,
                sync_dist=True,
                on_epoch=True,
            )

        with torch.inference_mode():
            preds = self.model.forward_to_proposals(batch["image"])
            score_maps = self.model.forward_to_heatmaps(batch["image"])
            score_maps = [x.mul(255).to(torch.uint8) for x in score_maps]
            # nme = compute_nme(batch['boxes'], batch['lmk5pts'], batch['has_lmk5pts'], preds[0], preds[2])
            # self.log(f'{mode}_nme', nme, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, logger=True)

        outputs = {
            "loss": loss,
            "img": batch["image"].permute(0, 2, 3, 1),
            "score_maps": score_maps,
            "gt_boxes": batch["boxes"],
            "gt_lmk5pts": batch["lmk5pts"],
            "gt_has_lmk5pts": batch["has_lmk5pts"],
            "gt_boxes_area": batch["boxes_area"],
            "mean": batch["mean"],
            "std": batch["std"],
            "pred_boxes": preds[0],
            "pred_objs": preds[1],
            "pred_lmk5pts": preds[2],
            "batch_idx": batch_idx,
            "gpu_id": batch["image"].device.index,
            # 'nme': nme,
        }
        if self.cfg["show_detail_every_step"][mode] > random.random():
            self._plot_images(mode, outputs)
        return outputs

    def training_step(self, batch, batch_idx, *args, **kwargs):
        opts = self.optimizers()
        opts = [opts] if not isinstance(opts, list) else opts
        for i, opt in enumerate(opts):
            lr = opt.param_groups[-1]["lr"]
            self.log(f"lr_{i}", lr, on_step=True, logger=False, prog_bar=True, sync_dist=True)
        outs = self._step("train", batch, batch_idx)
        # self.training_boxes_area.append(torch.cat(outs['gt_boxes_area']))
        return {"loss": outs["loss"]}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        outs = self._step("valid", batch, batch_idx)
        preds = [
            dict(
                boxes=boxes,
                scores=scores.flatten(),
                labels=torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device),
            )
            for boxes, scores in zip(outs["pred_boxes"], outs["pred_objs"])
        ]
        targets = [
            dict(
                boxes=boxes,
                labels=torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device),
            )
            for boxes in outs["gt_boxes"]
        ]
        self.metric(preds, targets)
        return {"loss": outs["loss"]}

    def _plot_images(self, mode, outputs):
        img = outputs["img"][0].cpu().numpy()
        mean, std = outputs["mean"][0].cpu().numpy(), outputs["std"][0].cpu().numpy()
        img = img * std + mean
        img = cb.imcvtcolor(img, "RGB2BGR")

        # plot gt
        plotted = np.concatenate((img, img), axis=1)
        gt_boxes = outputs["gt_boxes"][0].cpu().numpy()
        gt_scores = np.ones(gt_boxes.shape[0])
        gt_lmk5pts_list = outputs["gt_lmk5pts"][0].cpu().numpy().reshape(-1, 5, 2)
        # plot pred
        pred_boxes = outputs["pred_boxes"][0].float().cpu().numpy() + (640, 0, 640, 0)
        pred_objs = outputs["pred_objs"][0].float().cpu().numpy()
        pred_lmk5pts_list = outputs["pred_lmk5pts"][0].float().cpu().numpy().reshape(-1, 5, 2) + (640, 0)

        boxes = cb.Boxes(np.concatenate((gt_boxes, pred_boxes), axis=0))
        scores = np.concatenate((gt_scores, pred_objs), axis=0)
        lmk5pts_list = cb.KeypointsList(np.concatenate((gt_lmk5pts_list, pred_lmk5pts_list), axis=0))

        plotted = draw_results(
            plotted,
            boxes=boxes,
            scores=scores,
            kpts_list=lmk5pts_list,
            show_score=False,
        )

        log_dir = "tmp" if getattr(self.logger, "log_dir", None) is None else self.logger.log_dir
        log_dir = cb.Path(log_dir)

        # save plotted to file and tensorboard
        fname = f"image-{mode}/{self.current_epoch}/{outputs['batch_idx']}_gpu-id={outputs['gpu_id']}"
        fpath = log_dir / f"{fname}.png"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        cb.imwrite(plotted, fpath)
        self.logger.experiment.add_image(
            fname,
            cb.imread(str(fpath), "RGB"),
            self.global_step,
            dataformats="HWC",
        )

        # plot anchors and score_maps
        score_maps = [x[0].cpu().numpy().astype("uint8") for x in outputs["score_maps"]]
        gt_boxes = outputs["gt_boxes"][0]
        plotted_assign = self.model.head.plot_assign_priors(img, gt_boxes, score_maps)
        # save
        fname = f"image-{mode}/{self.current_epoch}/{outputs['batch_idx']}_gpu-id={outputs['gpu_id']}_assign_priors"
        fpath = log_dir / f"{fname}.png"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        cb.imwrite(plotted_assign, fpath)
        self.logger.experiment.add_image(
            fname,
            cb.imread(str(fpath), "RGB"),
            self.global_step,
            dataformats="HWC",
        )

    # def on_validation_start(self):
    #     face_areas = torch.cat(self.training_boxes_area).cpu().numpy()
    #     df = pd.DataFrame(data=np.array(face_areas), columns=['Area'])
    #     fig = sns.displot(df, x="Area", kde=True, log_scale=(True, False), stat='probability').set(xlim=(1, 1e7)).figure
    #     fig.patch.set_facecolor('white')
    #     fig.suptitle(f"train")
    #     fig.tight_layout()

    #     # save plotted to file and tensorboard
    #     gpu_id = self.local_rank
    #     fname = f"image-train/{self.current_epoch}/train_face-areas_gpu-id={gpu_id}"
    #     fpath = cb.Path(self.logger.log_dir, f'{fname}.png')
    #     fpath.parent.mkdir(parents=True, exist_ok=True)

    #     fig.savefig(str(fpath))
    #     self.logger.experiment.add_image(
    #         fname,
    #         cb.imread(str(fpath), "RGB"),
    #         self.global_step,
    #         dataformats="HWC",
    #     )
    #     self.training_boxes_area = []

    def on_validation_epoch_end(self) -> None:
        metrics = self.metric.compute()
        metrics.pop("classes")
        metrics = {k: v.to(device=self.device) for k, v in metrics.items()}
        mAP = metrics.pop("map")
        self.log("mAP", mAP, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, logger=True, sync_dist=True)
        self.metric.reset()

        tmp_onnx = "/tmp/scrfd.onnx"

        if self.cfg.get("widerface_evluation", None) is not None:
            if self.trainer.is_global_zero:
                self.to_onnx(tmp_onnx)
                widerface.main(
                    onnx_fpath=tmp_onnx,
                    **self.cfg["widerface_evluation"],
                )
            self.trainer.strategy.barrier()

        if self.cfg.get("nist_evaluation", None) is not None:
            if self.trainer.is_global_zero:
                self.to_onnx(tmp_onnx)
                nist.main(
                    onnx_fpath=tmp_onnx,
                    **self.cfg["nist_evaluation"],
                )
            self.trainer.strategy.barrier()

    def to_onnx(self, onnx_path: str, **kwargs):  # , do_quant: bool = False
        if self._trainer is not None:
            if self.trainer.is_global_zero:
                onnx_path = self.model.to_onnx(
                    onnx_path,
                    **kwargs,
                )
            self.trainer.strategy.barrier()
        else:
            onnx_path = self.model.to_onnx(
                onnx_path,
                **kwargs,
            )
        return onnx_path
