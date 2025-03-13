from typing import Tuple

import torch

from detector.head.scrfd_head import (AnchorGenerator, ATSSAssigner,
                                     InstanceData, PseudoSampler,
                                     anchor_inside_flags, bbox_overlaps, unmap)


def get_num_level_priors_inside(num_level_priors, inside_flags):
    split_inside_flags = torch.split(inside_flags, num_level_priors)
    num_level_priors_inside = [int(flags.sum()) for flags in split_inside_flags]
    return num_level_priors_inside


prior_generator = AnchorGenerator(
    scales=[1.0, 2.0],
    ratios=[1.0],
    base_sizes=[16, 64, 256],
    strides=[8, 16, 32]
)
assigner = ATSSAssigner(topk=9)
sampler = PseudoSampler()


def get_priors(
    feat_sizes,
    img_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    priors = [x for x in prior_generator.grid_priors(feat_sizes, device='cpu')]
    valid_flag = [x for x in prior_generator.valid_flags(feat_sizes, img_size, device='cpu')]
    return priors, valid_flag


def gen_one_targets(
    img_size: Tuple[int, int],
    loc_gts: torch.Tensor,
    lmk5pt_gts: torch.Tensor,
    has_lmk5pt_gts: torch.Tensor,
) -> tuple:
    h, w = img_size
    feat_sizes = [(int(h / stride[0]), int(w / stride[1])) for stride in prior_generator.strides]
    priors, valid_flags = get_priors(feat_sizes, img_size)
    num_level_priors = [len(x) for x in priors]
    flat_priors = torch.cat(priors).type_as(loc_gts)
    flat_valid_flags = torch.cat(valid_flags)

    inside_flags = anchor_inside_flags(
        flat_priors,
        flat_valid_flags,
        img_size,
        allowed_border=-1
    )

    if inside_flags.any():
        # assign gt and sample priors
        flat_priors = flat_priors[inside_flags]
        pred_instances = InstanceData(metainfo={'priors': flat_priors})
        num_level_priors_inside = get_num_level_priors_inside(num_level_priors, inside_flags)
        labels = torch.ones(len(loc_gts), dtype=torch.long, device=loc_gts.device)
        gt_instances = InstanceData(metainfo={'bboxes': loc_gts, 'labels': labels})
        assign_result = assigner.assign(
            pred_instances=pred_instances,
            num_level_priors=num_level_priors_inside,
            gt_instances=gt_instances,
        )
        sampling_result = sampler.sample(
            assign_result,
            pred_instances,
            gt_instances,
        )

        loc_targets = torch.zeros((len(flat_priors), 4), dtype=loc_gts.dtype, device=loc_gts.device)
        lmk5pt_targets = torch.zeros((len(flat_priors), 10), dtype=lmk5pt_gts.dtype, device=lmk5pt_gts.device)
        bg_targets = torch.full((len(flat_priors), ), -1, dtype=torch.long, device=lmk5pt_gts.device)

        obj_weights = torch.zeros((len(flat_priors),), dtype=loc_gts.dtype, device=loc_gts.device)
        lmk5pt_weights = torch.zeros((len(flat_priors),), dtype=lmk5pt_gts.dtype, device=lmk5pt_gts.device)

        pos_priors = sampling_result.pos_priors
        assign_pos_inds = sampling_result.pos_assigned_gt_inds
        pos_gt_anchor_ious = bbox_overlaps(loc_gts[assign_pos_inds], pos_priors, is_aligned=True)
        neg_inds = sampling_result.neg_inds
        pos_inds = sampling_result.pos_inds
        is_pos_mask = pos_gt_anchor_ious >= 0
        neg_inds = torch.concat((neg_inds, pos_inds[~is_pos_mask]), dim=0)
        pos_inds = pos_inds[is_pos_mask]
        assign_pos_inds = assign_pos_inds[is_pos_mask]
        if len(pos_inds):
            loc_targets[pos_inds] = loc_gts[assign_pos_inds]
            lmk5pt_targets[pos_inds] = lmk5pt_gts[assign_pos_inds]
            bg_targets[pos_inds] = 0
            lmk5pt_weights[pos_inds] = has_lmk5pt_gts[assign_pos_inds]
            obj_weights[pos_inds] = 1.0

        if len(neg_inds):
            obj_weights[neg_inds] = 1.0

        # map up to original set of priors
        num_total_priors = len(flat_priors)
        flat_priors = unmap(flat_priors, num_total_priors, inside_flags)
        loc_targets = unmap(loc_targets, num_total_priors, inside_flags)
        lmk5pt_targets = unmap(lmk5pt_targets, num_total_priors, inside_flags)
        bg_targets = unmap(bg_targets, num_total_priors, inside_flags, fill=1)
        obj_weights = unmap(obj_weights, num_total_priors, inside_flags)
        lmk5pt_weights = unmap(lmk5pt_weights, num_total_priors, inside_flags)
        return (
            flat_priors,
            loc_targets,
            lmk5pt_targets,
            bg_targets,
            lmk5pt_weights,
            obj_weights,
            pos_inds,
            neg_inds,
            num_level_priors,
        )


def get_target_single(
    img_size,
    gt_bboxes,
    gt_bboxes_ignore,
    gt_labels,
    gt_keypointss,
):

    h, w = img_size
    feat_sizes = [(int(h / stride[0]), int(w / stride[1])) for stride in prior_generator.strides]
    priors, valid_flags = get_priors(feat_sizes, img_size)
    num_level_priors = [len(x) for x in priors]
    flat_priors = torch.cat(priors).type_as(gt_bboxes)
    flat_valid_flags = torch.cat(valid_flags)

    inside_flags = anchor_inside_flags(
        flat_priors,
        flat_valid_flags,
        img_size,
        allowed_border=-1
    )
    num_level_priors_inside = get_num_level_priors_inside(num_level_priors, inside_flags)

    labels = torch.ones(len(gt_bboxes), dtype=torch.long, device=gt_bboxes.device)
    pred_instances = InstanceData(metainfo={'priors': flat_priors})
    gt_instances = InstanceData(metainfo={'bboxes': gt_bboxes, 'labels': labels})
    gt_instances_ignore = InstanceData(metainfo={'bboxes': gt_bboxes_ignore, 'labels': labels})
    assign_result = assigner.assign(
        pred_instances=pred_instances,
        num_level_priors=num_level_priors_inside,
        gt_instances=gt_instances,
        gt_instances_ignore=gt_instances_ignore,
    )
    sampling_result = sampler.sample(
        assign_result,
        pred_instances,
        gt_instances,
    )

    num_valid_priors = flat_priors.shape[0]
    bbox_targets = torch.zeros_like(flat_priors)
    bbox_weights = torch.zeros_like(flat_priors)
    kps_targets = flat_priors.new_zeros(size=(flat_priors.shape[0], 10))
    kps_weights = flat_priors.new_zeros(size=(flat_priors.shape[0], 10))
    labels = flat_priors.new_full((num_valid_priors, ), -1, dtype=torch.long)
    label_weights = flat_priors.new_zeros(num_valid_priors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = sampling_result.pos_gt_bboxes
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        # print('BBB', flat_priors.shape, gt_bboxes.shape, gt_keypointss.shape, pos_inds.shape, bbox_targets.shape, pos_bbox_targets.shape)
        kps_targets[pos_inds, :] = gt_keypointss[pos_assigned_gt_inds, :, :2].reshape((-1, 10))
        kps_weights[pos_inds, :] = torch.mean(gt_keypointss[pos_assigned_gt_inds, :, 2], dim=1, keepdims=True)
        # kps_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            # Only rpn gives gt_labels as None
            # Foreground is the first class
            labels[pos_inds] = 0
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        label_weights[pos_inds] = 1.0

    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of flat_priors
    num_total_priors = flat_priors.size(0)
    flat_priors = unmap(flat_priors, num_total_priors, inside_flags)
    labels = unmap(labels, num_total_priors, inside_flags, fill=1)
    label_weights = unmap(label_weights, num_total_priors,
                          inside_flags)
    bbox_targets = unmap(bbox_targets, num_total_priors, inside_flags)
    bbox_weights = unmap(bbox_weights, num_total_priors, inside_flags)
    kps_targets = unmap(kps_targets, num_total_priors, inside_flags)
    kps_weights = unmap(kps_weights, num_total_priors, inside_flags)

    return (flat_priors, labels, label_weights, bbox_targets, bbox_weights,
            kps_targets, kps_weights, pos_inds, neg_inds)


def main():
    img_size = (640, 640)
    loc_gts = torch.zeros((1, 4), dtype=torch.float32)
    lmk5pt_gts = torch.ones((1, 5, 3), dtype=torch.float32)
    has_lmk5pt_gts = torch.ones((1, ), dtype=torch.float32)
    loc_gts[0] = torch.tensor([0, 0, 100, 100])
    lmk5pt_gts[..., :2] = torch.tensor([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
    target1 = gen_one_targets(img_size, loc_gts, lmk5pt_gts[..., :2].reshape(-1, 10), has_lmk5pt_gts)
    target2 = get_target_single(img_size, loc_gts, loc_gts, None, lmk5pt_gts)

    breakpoint()


main()
