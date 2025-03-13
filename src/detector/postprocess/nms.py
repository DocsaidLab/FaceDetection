from typing import List, Tuple, Union

import numpy as np
import torch
from torchvision.ops import nms as torch_nms


def np2tensor(x):
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    return x


def tensor2np(x):
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    return x


@ torch.inference_mode()
def do_nms(
    decoded_pred: Union[Tuple[torch.Tensor, ...], Tuple[np.ndarray, ...]],
    score_thres: float = 0.02,
    nms_topk: int = 5000,
    nms_th: float = 0.45,
) -> Tuple[torch.Tensor, ...]:
    # loc, obj, lmk5pt, pose, cls = decoded_pred
    loc, obj, lmk5pt = decoded_pred
    obj = obj.flatten()

    loc = np2tensor(loc)
    obj = np2tensor(obj)
    lmk5pt = np2tensor(lmk5pt)

    # filter out bboxes are not pass the threshold
    valid = (obj > score_thres).flatten()
    loc = loc[valid]
    obj = obj[valid]
    lmk5pt = lmk5pt[valid]

    # sorted boxes by scores
    inds = obj.argsort(0, descending=True)
    loc = loc[inds]
    obj = obj[inds]
    lmk5pt = lmk5pt[inds]

    # run nms and keep top k nms results
    keep = torch_nms(loc.float(), obj.float(), nms_th)[:nms_topk]
    loc = loc[keep]
    obj = obj[keep]
    lmk5pt = lmk5pt[keep]

    # order scores
    return loc, obj, lmk5pt


@ torch.inference_mode()
def do_batch_nms(
    decoded_preds: Union[Tuple[torch.Tensor, ...], Tuple[np.ndarray, ...]],
    score_th: float = 0.02,
    nms_topk: int = 5000,
    nms_th: float = 0.45,
) -> Tuple[List[torch.Tensor], ...]:
    # keep top k predictions
    out_locs, out_objs, out_lmk5pts = [], [], []
    for preds in zip(*decoded_preds):
        loc, obj, lmk5pt = do_nms(preds, score_th, nms_topk, nms_th)
        out_locs.append(loc)
        out_objs.append(obj)
        out_lmk5pts.append(lmk5pt)
    return out_locs, out_objs, out_lmk5pts


def nms(dets, nms_thresh):
    thresh = nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
