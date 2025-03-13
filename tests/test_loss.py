from typing import Tuple

import torch

from detector.head.scrfd_head import (AnchorGenerator, DIoULoss,
                                     QualityFocalLoss, SmoothL1Loss,
                                     bbox_overlaps, distance2bbox,
                                     get_prior_center, kps2distance)

loc_loss = DIoULoss(loss_weight=2)
obj_loss = QualityFocalLoss(
    use_sigmoid=True,
    beta=2.0,
    loss_weight=1,
)
lmk5pt_loss = SmoothL1Loss(
    beta=1. / 9,
    loss_weight=0.1,
)

prior_generator = AnchorGenerator(
    scales=[1.0, 2.0],
    ratios=[1.0],
    base_sizes=[16],
    strides=[8]
)


def anchor_center(anchors):
    """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
    anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
    anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
    return torch.stack([anchors_cx, anchors_cy], dim=-1)


def calc_level_loss(
    loc_preds: torch.Tensor,
    obj_preds: torch.Tensor,
    lmk5pt_preds: torch.Tensor,
    loc_targets: torch.Tensor,
    lmk5pt_targets: torch.Tensor,
    bg_targets: torch.Tensor,
    obj_weights: torch.Tensor,
    lmk5pt_weights: torch.Tensor,
    flat_priors: torch.Tensor,
    stride: Tuple[int, int],
    num_total_samples: int,
):
    if stride[0] != stride[1]:
        raise ValueError('h stride is not equal to w stride!')

    loc_preds = loc_preds.float().permute(0, 2, 3, 1).reshape(-1, 4)
    obj_preds = obj_preds.float().permute(0, 2, 3, 1).reshape(-1, 1)
    lmk5pt_preds = lmk5pt_preds.float().permute(0, 2, 3, 1).reshape(-1, 10)
    loc_targets = loc_targets.reshape(-1, 4).type_as(loc_preds)
    lmk5pt_targets = lmk5pt_targets.reshape(-1, 10).type_as(lmk5pt_preds)
    bg_targets = bg_targets.reshape(-1)

    obj_weights = obj_weights.reshape(-1).type_as(obj_preds)
    lmk5pt_weights = lmk5pt_weights.reshape(-1).type_as(lmk5pt_preds)

    flat_priors = flat_priors.reshape(-1, 4).type_as(loc_preds)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    pos_inds = torch.argwhere(bg_targets == 0).flatten()
    obj_scores = torch.zeros_like(bg_targets).type_as(obj_preds)

    if len(pos_inds):
        level_stride = stride[0]
        pos_prior_centers = get_prior_center(flat_priors[pos_inds])
        pos_prior_centers = pos_prior_centers / level_stride

        pos_loc_level_targets = loc_targets[pos_inds] / level_stride
        pos_loc_distance_preds = loc_preds[pos_inds]
        pos_loc_level_preds = distance2bbox(pos_prior_centers, pos_loc_distance_preds)

        pos_loc_weights = obj_preds.detach().sigmoid()
        pos_loc_weights = pos_loc_weights.max(dim=1)[0][pos_inds]
        obj_scores[pos_inds] = bbox_overlaps(
            pos_loc_level_preds.detach(),
            pos_loc_level_targets,
            is_aligned=True,
        )

        pos_lmk5pt_level_targets = lmk5pt_targets[pos_inds].flatten(start_dim=1) / level_stride
        pos_lmk5pt_distance_targets = kps2distance(pos_prior_centers, pos_lmk5pt_level_targets)
        pos_lmk5pt_distance_preds = lmk5pt_preds[pos_inds]
        pos_lmk5pt_weights = lmk5pt_weights[pos_inds] * pos_loc_weights
        pos_lmk5pt_weights = pos_lmk5pt_weights[..., None]

        lloc = loc_loss(
            pos_loc_level_preds,
            pos_loc_level_targets,
            weight=pos_loc_weights,
            avg_factor=1.0,
        )
        llmk5pt = lmk5pt_loss(
            pos_lmk5pt_distance_preds,
            pos_lmk5pt_distance_targets,
            weight=pos_lmk5pt_weights,
            avg_factor=1.0,
        )
    else:
        lloc = loc_preds.sum() * 0
        lobj = torch.zeros_like(lloc)
        llmk5pt = torch.zeros_like(lloc)
        pos_loc_weights = torch.zeros_like(lloc)

    lobj = obj_loss(
        obj_preds, (bg_targets, obj_scores),
        weight=obj_weights,
        avg_factor=num_total_samples
    )

    return lloc, lobj, llmk5pt, pos_loc_weights.sum()


def loss_single(
    bbox_pred, cls_score, kps_pred, bbox_targets, kps_targets, labels, label_weights,
    kps_weights, anchors, stride, num_total_samples
):
    anchors = anchors.reshape(-1, 4)
    cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, 1)
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    bbox_targets = bbox_targets.reshape(-1, 4)
    labels = labels.reshape(-1)
    label_weights = label_weights.reshape(-1)

    kps_pred = kps_pred.permute(0, 2, 3, 1).reshape(-1, 10)
    kps_targets = kps_targets.reshape(-1, 10)
    kps_weights = kps_weights.reshape(-1, 10)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = 1
    pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
    score = label_weights.new_zeros(labels.shape)

    if len(pos_inds):
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_pred = bbox_pred[pos_inds]
        pos_anchors = anchors[pos_inds]
        pos_anchor_centers = anchor_center(pos_anchors) / stride[0]

        weight_targets = cls_score.detach().sigmoid()
        weight_targets = weight_targets.max(dim=1)[0][pos_inds]
        pos_decode_bbox_targets = pos_bbox_targets / stride[0]
        pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred)

        pos_kps_targets = kps_targets[pos_inds]
        pos_kps_pred = kps_pred[pos_inds]
        pos_kps_weights = kps_weights.max(dim=1)[0][pos_inds] * weight_targets
        pos_kps_weights = pos_kps_weights.reshape((-1, 1))

        pos_decode_kps_targets = kps2distance(pos_anchor_centers, pos_kps_targets / stride[0])
        pos_decode_kps_pred = pos_kps_pred

        score[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True,
        )

        # regression loss
        loss_bbox = loc_loss(
            pos_decode_bbox_pred,
            pos_decode_bbox_targets,
            weight=weight_targets,
            avg_factor=1.0,
        )

        loss_kps = lmk5pt_loss(
            pos_decode_kps_pred,
            pos_decode_kps_targets,
            weight=pos_kps_weights,
            avg_factor=1.0,
        )

    else:
        loss_bbox = bbox_pred.sum() * 0
        loss_kps = kps_pred.sum() * 0
        weight_targets = torch.zeros_like(loss_kps)

    loss_cls = obj_loss(
        cls_score, (labels, score),
        weight=label_weights,
        avg_factor=num_total_samples,
    )

    return loss_bbox, loss_cls, loss_kps, weight_targets.sum()


def main():
    loc_preds = torch.zeros((1, 40, 40, 4), dtype=torch.float32)
    obj_preds = torch.full((1, 40, 40, 1), fill_value=-4.925)
    lmk5pt_preds = torch.zeros((1, 40, 40, 10), dtype=torch.float32)
    loc_targets: torch.Tensor = torch.zeros((1, 40, 40, 4), dtype=torch.float32)
    lmk5pt_targets: torch.Tensor = torch.zeros((1, 40, 40, 10), dtype=torch.float32)
    bg_targets: torch.Tensor = torch.full((1, 40, 40), fill_value=-1, dtype=torch.float32)
    obj_weights: torch.Tensor = torch.ones((1, 40, 40), dtype=torch.float32)
    lmk5pt_weights: torch.Tensor = torch.ones((1, 40, 40), dtype=torch.float32)
    flat_priors: torch.Tensor = prior_generator.grid_priors([(40, 40)], device=loc_preds.device)[0]
    stride: Tuple[int, int] = (8, 8)
    num_total_samples: int = 9

    bg_targets[0, 0, 0] = 0
    obj_preds[0, 0, 0] = 0
    loc_preds[0, 0, 0] = torch.tensor([-0.08, -0.09, 0.19, 0.17])
    loc_targets[0, 0, 0] = torch.tensor([-0.1, -0.1, 0.2, 0.2])
    lmk5pt_preds[0, 0, 0] = torch.tensor([0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
    lmk5pt_targets[0, 0, 0] = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5])

    loss1 = calc_level_loss(
        loc_preds,
        obj_preds,
        lmk5pt_preds,
        loc_targets,
        lmk5pt_targets,
        bg_targets,
        obj_weights,
        lmk5pt_weights,
        flat_priors,
        stride,
        num_total_samples,
    )
    loss2 = loss_single(
        loc_preds,
        obj_preds,
        lmk5pt_preds,
        loc_targets,
        lmk5pt_targets,
        bg_targets,
        obj_weights,
        lmk5pt_weights,
        flat_priors,
        stride,
        num_total_samples,
    )
    print(loss1)
    print(loss2)
    breakpoint()


main()
