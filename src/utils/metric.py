from typing import List

import torch
import torchmetrics as tm
from mmdet.structures.bbox import bbox_overlaps
from torchmetrics.detection.mean_ap import MeanAveragePrecision as _MAP
from torchvision.ops import box_iou


def _get_normalized_area(boxes: torch.Tensor):
    hs = boxes[:, 3] - boxes[:, 1]
    ws = boxes[:, 2] - boxes[:, 0]
    return (hs * ws) ** 0.5


def compute_nme(batch_gt_boxes, batch_gt_lmk5pts, batch_has_lmk5pts, batch_pred_boxes, batch_pred_lmk5pts):
    '''
    1. get matched gt_box and pred_box
    2. calculate normalized area
    '''
    zipped = zip(batch_gt_boxes, batch_gt_lmk5pts, batch_has_lmk5pts, batch_pred_boxes, batch_pred_lmk5pts)
    nme = torch.zeros(1).type_as(batch_gt_boxes[0])
    n_elements = 0 + 1e-8
    for gt_boxes, gt_lmk5pts, has_lmk5pts, pred_boxes, pred_lmk5pts in zipped:
        has_lmk5pts = torch.argwhere(has_lmk5pts).flatten()
        gt_boxes = gt_boxes[has_lmk5pts]
        gt_lmk5pts = gt_lmk5pts[has_lmk5pts]
        tmp_nme = torch.full_like(gt_lmk5pts, 1)
        if len(gt_boxes) and len(pred_boxes):
            iou_matrix = box_iou(gt_boxes, pred_boxes)
            max_ious, choose_inds = iou_matrix.max(dim=1)
            mask = max_ious >= 0.85
            if mask.sum():
                gt_boxes = gt_boxes[mask]
                gt_lmk5pts = gt_lmk5pts[mask]
                choose_inds = choose_inds[mask]
                pred_lmk5pts = pred_lmk5pts[choose_inds]
                normalized_areas = _get_normalized_area(gt_boxes)
                preds = pred_lmk5pts.reshape(-1, 10) / (normalized_areas[:, None] + 1e-8)
                targets = gt_lmk5pts.reshape(-1, 10) / (normalized_areas[:, None] + 1e-8)
                tmp_nme[mask] = (preds - targets).abs().clip(0, 1)
        nme += tmp_nme.sum()
        n_elements += tmp_nme.numel()
    return nme.div(n_elements)


def normalize_lmks_preds_and_gts(boxes: torch.Tensor, lmks: torch.Tensor, gts: torch.Tensor):
    boxes_gts = gts[..., :4]
    ious = box_iou(boxes, boxes_gts)
    max_v, pair_inds = ious.max(0)
    mask = max_v.flatten() > 0.7

    # initialize norm_lmks_gts and norm_lmks_preds with 0 and 1
    lmks_gts = gts[..., 5:]
    norm_lmks_gts = torch.zeros_like(lmks_gts)
    norm_lmks_preds = torch.ones_like(lmks_gts)
    n_matched = mask.sum()

    if n_matched:
        pair_inds = pair_inds[mask]
        lmks_gts = lmks_gts[mask]
        boxes_gts = boxes_gts[mask]
        lmks_preds = lmks[pair_inds]

        # For the same coorinate system for normalized computing
        # It must use (x1, y1, x2, y2) of targets and preds,

        x1, y1, x2, y2 = boxes_gts.unsqueeze(0).T
        cx, cy, w, h = (x2 + x1) / 2, (y2 + y1) / 2, (x2 - x1), (y2 - y1)
        lmks_gts[..., ::2] = (lmks_gts[..., ::2] - cx) / w
        lmks_gts[..., 1::2] = (lmks_gts[..., 1::2] - cy) / h
        norm_lmks_gts[:n_matched] = lmks_gts

        lmks_preds[..., ::2] = (lmks_preds[..., ::2] - cx) / w
        lmks_preds[..., 1::2] = (lmks_preds[..., 1::2] - cy) / h
        norm_lmks_preds[:n_matched] = lmks_preds

    return norm_lmks_preds, norm_lmks_gts


class NMSE(tm.MeanSquaredError):
    def __init__(self, overlap_ratio=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_ratio = overlap_ratio

    def normalize_lmks(
        self,
        pred_locs: torch.Tensor,
        pred_lmks: torch.Tensor,
        gt_locs: torch.Tensor,
        gt_lmks: torch.Tensor,
    ):
        ious = box_iou(pred_locs, gt_locs)  # N x 4, M x 4 -> N x M
        _, p2g_max_inds = ious.max(0)  # N x M -> 1 x M
        _, g2p_max_inds = ious.max(1)  # N x M -> N x 1
        matched_gt_inds = g2p_max_inds.unique()
        matched_pred_inds = p2g_max_inds[matched_gt_inds]

        gt_locs = gt_locs[matched_gt_inds]
        gt_lmks = gt_lmks[matched_gt_inds]
        pred_locs = pred_locs[matched_pred_inds]
        pred_lmks = pred_lmks[matched_pred_inds]
        ious = bbox_overlaps(pred_locs, gt_locs, is_aligned=True)
        mask = torch.argwhere(ious >= self.overlap_ratio).flatten()

        # initialize norm_glmks and norm_plmks with 0 and 1
        norm_glmks = torch.zeros_like(gt_lmks)
        norm_plmks = torch.ones_like(gt_lmks)
        n_matched = mask.sum()

        if n_matched:
            gt_lmks = gt_lmks[mask]
            gt_locs = gt_locs[mask]
            pred_lmks = pred_lmks[mask]

            # For the same coorinate system for normalized computing
            # It must use (x1, y1, x2, y2) of targets and preds,

            x1, y1, x2, y2 = gt_locs.T.unsqueeze(-1)
            cx, cy, w, h = (x2 + x1) / 2, (y2 + y1) / 2, (x2 - x1), (y2 - y1)
            gt_lmks[..., ::2] = (gt_lmks[..., ::2] - cx) / w
            gt_lmks[..., 1::2] = (gt_lmks[..., 1::2] - cy) / h
            norm_glmks[mask] = gt_lmks

            pred_lmks[..., ::2] = (pred_lmks[..., ::2] - cx) / w
            pred_lmks[..., 1::2] = (pred_lmks[..., 1::2] - cy) / h
            norm_plmks[mask] = pred_lmks.type_as(gt_lmks)

        return norm_plmks, norm_glmks

    def update(
        self,
        batch_plocs: List[torch.Tensor],
        batch_plmks: List[torch.Tensor],
        batch_glocs: List[torch.Tensor],
        batch_glmks: List[torch.Tensor],
        batch_has_lmks: List[torch.Tensor],
    ):
        batched_norm_plmks = []
        batched_norm_glmks = []
        zipped = zip(batch_plocs, batch_plmks, batch_glocs, batch_glmks, batch_has_lmks)
        for pred_locs, pred_lmks, gt_locs, gt_lmks, has_lmks in zipped:
            inds = torch.argwhere(has_lmks).flatten()
            gt_locs = gt_locs[inds]
            gt_lmks = gt_lmks[inds]
            if len(inds) * len(pred_locs) * len(pred_lmks):
                norm_plmks, norm_glmks = self.normalize_lmks(pred_locs, pred_lmks, gt_locs, gt_lmks)
                batched_norm_plmks.append(norm_glmks)
                batched_norm_glmks.append(norm_plmks)

        if len(batched_norm_plmks):
            batched_norm_plmks = torch.concat(batched_norm_plmks, 0)
            batched_norm_glmks = torch.concat(batched_norm_glmks, 0)
            super().update(batched_norm_plmks, batched_norm_glmks)

    def compute(self):
        results = super().compute()
        self.reset()
        return results


class MAP(_MAP):

    def update(self, batch_plocs, batch_pobjs, batch_glocs):
        preds = [
            dict(
                boxes=loc_preds,
                scores=obj_preds,
                labels=torch.zeros_like(obj_preds).long(),
            )
            for loc_preds, obj_preds in zip(batch_plocs, batch_pobjs)
        ]
        targets = [
            dict(
                boxes=loc_gts,
                labels=torch.zeros(len(loc_gts)).type_as(loc_gts).long(),
            )
            for loc_gts in batch_glocs
        ]
        super().update(preds, targets)

    def compute(self):
        results = super().compute()
        self.reset()
        return results


metrics_mapper = {
    'Accuracy': tm.Accuracy,
    'AUC': tm.AUROC,
    'ROC': tm.ROC,
    'ConfusionMatrix': tm.ConfusionMatrix,
    'NMSE': NMSE,
    'MAP': MAP,
}


def build_metrics(name, **kwargs):
    metric = metrics_mapper.get(name, None)
    if metric is None:
        raise ValueError(f'There are not support {name} model.')
    return metric(**kwargs)
