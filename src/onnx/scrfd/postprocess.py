from typing import Dict, List, Tuple

import numpy as np

from .numpy_nms import nms


def gen_prior_centers(
    feat_sizes: List[Tuple[int, int]],
    anc_strides: List[Tuple[int, int]],
    anc_img_scales: List[Tuple[int, int]],
):
    prior_centers = []
    for feat_size, stride in zip(feat_sizes, anc_strides):
        xx, yy = np.meshgrid(range(feat_size[1]), range(feat_size[0]))
        prior_grids = (
            np.stack((xx, yy), -1).reshape(-1, 2).repeat(len(anc_img_scales), axis=0)
        )
        prior_grids *= stride
        prior_centers.append(prior_grids)
    prior_centers = np.concatenate(prior_centers, axis=0)
    return prior_centers


def distance2bbox(points, distance) -> np.ndarray:
    bboxes = np.tile(points, 2) + distance * [-1, -1, 1, 1]
    return bboxes


def distance2kps(points, distance) -> np.ndarray:
    kpts = distance + np.tile(points, 5)
    return kpts


def get_flat_preds(preds: Dict[str, np.ndarray], strides: List[int]):
    level_loc_preds = []
    level_obj_preds = []
    level_lmk_preds = []
    for stride in strides:
        box_pred = preds[f"box_{stride}"]  # * stride
        score_pred = preds[f"score_{stride}"]
        lmk_pred = preds[f"lmk5pt_{stride}"]  # * stride
        # b = box_pred.shape[0]
        # box_pred = box_pred.reshape(b, -1, 4)
        # score_pred = score_pred.reshape(b, -1, 1)
        # lmk_pred = lmk_pred.reshape(b, -1, 10)

        level_loc_preds.append(box_pred)
        level_obj_preds.append(score_pred)
        level_lmk_preds.append(lmk_pred)

    flat_loc_preds = np.concatenate(level_loc_preds, 1)
    flat_obj_preds = np.concatenate(level_obj_preds, 1)
    flat_lmk_preds = np.concatenate(level_lmk_preds, 1)
    return flat_loc_preds, flat_obj_preds, flat_lmk_preds


def get_proposals(
    flat_preds: Tuple[np.ndarray, np.ndarray, np.ndarray],
    prior_centers: np.ndarray,
    img_scales: List[float],
    score_th: float = 0.02,
    nms_th: float = 0.45,
    nms_topk: int = 5000,
) -> List[Tuple[np.ndarray, ...]]:
    proposals_list = []
    for loc, obj, lmk, img_scale in zip(*flat_preds, img_scales):
        loc = distance2bbox(prior_centers, loc)
        lmk = distance2kps(prior_centers, lmk)

        valid = np.where(obj > score_th)[0]
        loc = loc[valid]
        obj = obj[valid]
        lmk = lmk[valid]

        if len(loc):
            dets = np.concatenate((loc, obj), axis=-1).astype("float32")
            keep = nms(dets, thresh=nms_th)[:nms_topk]

            loc = loc[keep]
            obj = obj[keep]
            lmk = lmk[keep].reshape(-1, 5, 2)

        proposals_list.append(
            {
                "boxes": loc / img_scale,
                "scores": obj,
                "lmk5pts": lmk / img_scale,
            }
        )
    return proposals_list
