from typing import Any, Dict, List, Optional, Tuple

import capybara as cb
import cv2
import numpy as np
import torch
import torch.nn as nn
from mmdet.models import DIoULoss, QualityFocalLoss, SmoothL1Loss
from mmdet.models.task_modules import (AnchorGenerator, ATSSAssigner,
                                       PseudoSampler, anchor_inside_flags)
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.structures.bbox import bbox_overlaps, distance2bbox
from mmdet.structures.det_data_sample import InstanceData
from mmdet.utils.dist_utils import reduce_mean

from ..postprocess import do_batch_nms
from ._components import ComponentHead, ComponentHeadFeature
from .base_head import BaseHead


def distance2kps(points, distance, n_points=5):
    flag = distance.dim() == 2
    distance = distance[None] if flag else distance
    outs = []
    for x in distance:
        outs.append(x + points.repeat_interleave(n_points, dim=0).reshape(*x.shape))
    outs = torch.stack(outs) if not flag else outs[0]
    return outs


def kps2distance(points, kps, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        kps (Tensor): Shape (n, K), "xy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """

    preds = []
    for i in range(0, kps.shape[-1], 2):
        px = kps[:, i] - points[:, i % 2]
        py = kps[:, i + 1] - points[:, i % 2 + 1]
        if max_dis is not None:
            px = px.clamp(min=0, max=max_dis - eps)
            py = py.clamp(min=0, max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)


def get_prior_center(priors):
    """Get prior centers from priors.

    Args:
        priors (Tensor): prior list with shape (N, 4), "xyxy" format.

    Returns:
        Tensor: prior centers with shape (N, 2), "xy" format.
    """
    priors_cx = (priors[..., 2] + priors[..., 0]) / 2
    priors_cy = (priors[..., 3] + priors[..., 1]) / 2
    return torch.stack([priors_cx, priors_cy], dim=-1)


class SCRFDHead(BaseHead):
    def setup_head(
        self,
    ):
        n_stack = self.head_cfg["n_stack"]
        n_levels = self.head_cfg["n_levels"]
        in_channels = self.head_cfg["in_channels"]
        hid_channels = self.head_cfg["hid_channels"]
        norm = self.head_cfg["norm"]
        act = self.head_cfg["act"]
        feat_share = self.head_cfg["feat_share"]
        use_scale = self.head_cfg["use_scale"]
        share_feats = []
        loc_feats = []
        lmk_feats = []
        loc_heads = []
        obj_heads = []
        lmk5pt_heads = []
        if isinstance(in_channels, int):
            in_channels = [in_channels] * n_levels

        if isinstance(hid_channels, int):
            hid_channels = [hid_channels] * n_levels

        if len(in_channels) != n_levels:
            raise ValueError("The length of incannels_list is not equal to n_levels.")

        n_priors = self.prior_generator.num_base_anchors[0]

        for inc, hidc in zip(in_channels, hid_channels):
            if feat_share:
                share_feats.append(ComponentHeadFeature(inc, hidc, n_stack, norm, act))
            else:
                loc_feats.append(ComponentHeadFeature(inc, hidc, n_stack, norm, act))
                lmk_feats.append(ComponentHeadFeature(inc, hidc, n_stack, norm, act))
            loc_heads.append(ComponentHead(hidc, n_priors, 4, use_scale=use_scale))
            obj_heads.append(ComponentHead(hidc, n_priors, 1, use_scale=False))
            lmk5pt_heads.append(ComponentHead(hidc, n_priors, 10, use_scale=False))

        if feat_share:
            self.share_feats = nn.ModuleList(share_feats)
        else:
            self.loc_feats = nn.ModuleList(loc_feats)
            self.lmk_feats = nn.ModuleList(lmk_feats)

        self.loc_heads = nn.ModuleList(loc_heads)
        self.obj_heads = nn.ModuleList(obj_heads)
        self.lmk5pt_heads = nn.ModuleList(lmk5pt_heads)
        self.n_levels = n_levels
        self.feat_share = feat_share
        self.init_weights()

    def init_weights(self):
        norm_group = (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)

        def _recursive_init(m, mean=0, std=0.01, bias=0):
            for child in m.children():
                if len(list(child.children())):
                    _recursive_init(child)
                else:
                    if isinstance(child, (nn.Conv2d, nn.Linear)):
                        nn.init.normal_(child.weight, mean, std)
                        if child.bias is not None:
                            nn.init.constant_(child.bias, bias)
                    elif isinstance(child, norm_group):
                        nn.init.constant_(child.weight, 1)
                        if child.bias is not None:
                            nn.init.constant_(child.bias, 0)

        """Initialize weights of the head."""
        if self.feat_share:
            for m in self.share_feats.children():
                _recursive_init(m, std=0.01)
        else:
            for m in self.loc_feats.children():
                _recursive_init(m, std=0.01)
            for m in self.lmk_feats.children():
                _recursive_init(m, std=0.01)

        for m in self.loc_heads.children():
            _recursive_init(m, std=0.01)
        for m in self.obj_heads.children():
            _recursive_init(m, std=0.01, bias=-4.595)
        for m in self.lmk5pt_heads.children():
            _recursive_init(m, std=0.01)

    def setup_prior(self):
        self.prior_generator = AnchorGenerator(**self.prior_cfg)
        self.prior_cfg = {
            "strides": [list(x) for x in self.prior_generator.strides],
            "sizes": [
                [size for s in self.prior_generator.scales]
                for size in self.prior_generator.base_sizes
            ],
        }

    def setup_loss(self):
        self.loc_loss = DIoULoss(loss_weight=self.loss_weight["loc"])
        self.obj_loss = QualityFocalLoss(
            use_sigmoid=True,
            beta=2.0,
            loss_weight=self.loss_weight["obj"],
        )
        self.lmk5pt_loss = SmoothL1Loss(
            beta=1.0 / 9,
            loss_weight=self.loss_weight["lmk5pt"],
        )
        self.assigner = ATSSAssigner(topk=9)
        self.sampler = PseudoSampler()

    def forward_preds(self, xs) -> Tuple[torch.Tensor, ...]:
        locs, objs, lmk5pts = [], [], []

        if self.share_feats is not None:
            zipped = zip(
                xs, self.share_feats, self.loc_heads, self.obj_heads, self.lmk5pt_heads
            )

            for x, share_head, mloc_head, mobj_head, mlmk5pt_head in zipped:
                head_feat = share_head(x)
                locs.append(mloc_head(head_feat))
                objs.append(mobj_head(head_feat))
                lmk5pts.append(mlmk5pt_head(head_feat))
        else:
            zipped = zip(
                xs,
                self.loc_feats,
                self.lmk_feats,
                self.loc_heads,
                self.obj_heads,
                self.lmk5pt_heads,
            )

            for x, mloc_feat, mlmk_feat, mloc_head, mobj_head, mlmk5pt_head in zipped:
                loc_feat = mloc_feat(x)
                lmk_feat = mlmk_feat(x)
                locs.append(mloc_head(loc_feat))
                objs.append(mobj_head(loc_feat))
                lmk5pts.append(mlmk5pt_head(lmk_feat))

        return locs, objs, lmk5pts

    def forward_train(
        self,
        feats: List[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        loc_gts_list = batch["boxes"]
        lmk5pt_gts_list = batch["lmk5pts"]
        has_lmk5pt_gts_list = batch["has_lmk5pts"]
        img_size = batch["image"].shape[-2:]
        preds = self.forward_preds(feats)
        losses = self._calc_loss(
            *preds,
            loc_gts_list=loc_gts_list,
            lmk5pt_gts_list=lmk5pt_gts_list,
            has_lmk5pt_gts_list=has_lmk5pt_gts_list,
            img_size=img_size,
        )
        return losses

    def forward(self, xs) -> List[Dict[str, torch.Tensor]]:
        outs = {}
        level_strides = self.prior_cfg["strides"]
        locs, objs, lmk5pts = self.forward_preds(xs)
        for loc, obj, lmk5pt, (stride, _) in zip(locs, objs, lmk5pts, level_strides):
            outs[f"box_{stride}"] = (
                loc.mul(stride).permute(0, 2, 3, 1).reshape(loc.shape[0], -1, 4)
            )
            outs[f"score_{stride}"] = (
                obj.sigmoid().permute(0, 2, 3, 1).reshape(obj.shape[0], -1, 1)
            )
            outs[f"lmk5pt_{stride}"] = (
                lmk5pt.mul(stride).permute(0, 2, 3, 1).reshape(lmk5pt.shape[0], -1, 10)
            )
        return outs

    @torch.inference_mode()
    def forward_to_proposals(
        self, feats: List[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], ...]:
        level_preds = self(feats)
        feat_sizes = [x.shape[2:] for x in feats[: self.n_levels]]
        level_priors = self.prior_generator.grid_priors(
            feat_sizes, dtype=feats[0].dtype, device=feats[0].device
        )
        strides = [x[0] for x in self.prior_generator.strides]
        outs = self._postprocess_to_proposals(
            level_preds, level_priors, strides, self.nms_cfg
        )
        return outs

    @torch.inference_mode()
    def forward_to_heatmaps(
        self, feats: List[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], ...]:
        _, score_maps, *_ = self.forward_preds(feats)
        score_maps = [x.sigmoid() for x in score_maps]
        return score_maps

    @torch.inference_mode()
    def get_priors(
        self,
        feat_sizes: List[Tuple[int, int]],
        img_size: Tuple[int, int],
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        priors = self.prior_generator.grid_priors(feat_sizes, device=device)
        valid_flag = self.prior_generator.valid_flags(
            feat_sizes, img_size, device=device
        )
        return priors, valid_flag

    @staticmethod
    @torch.inference_mode()
    def _postprocess_to_proposals(
        level_preds: Dict[str, torch.Tensor],
        level_priors: List[torch.Tensor],
        strides: List[int],
        nms_cfg: Optional[dict] = {},
    ):
        level_loc_preds = [level_preds[f"box_{stride}"] for stride in strides]
        level_obj_preds = [level_preds[f"score_{stride}"] for stride in strides]
        level_lmk5pt_preds = [level_preds[f"lmk5pt_{stride}"] for stride in strides]

        flat_loc_preds = torch.cat(level_loc_preds, dim=1)
        flat_obj_preds = torch.cat(level_obj_preds, dim=1)
        flat_lmk5pt_preds = torch.cat(level_lmk5pt_preds, dim=1)
        flat_priors = torch.cat(level_priors, dim=0)
        flat_prior_centers = get_prior_center(flat_priors)

        flat_loc_preds = distance2bbox(flat_prior_centers, flat_loc_preds)
        flat_lmk5pt_preds = distance2kps(flat_prior_centers, flat_lmk5pt_preds)
        results = flat_loc_preds, flat_obj_preds, flat_lmk5pt_preds
        return do_batch_nms(results, **nms_cfg)

    def _calc_loss(
        self,
        level_loc_preds: List[torch.Tensor],
        level_obj_preds: List[torch.Tensor],
        level_lmk5pt_preds: List[torch.Tensor],
        *,
        loc_gts_list: List[torch.Tensor],
        lmk5pt_gts_list: List[torch.Tensor],
        has_lmk5pt_gts_list: List[torch.Tensor],
        img_size: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        if len(level_loc_preds) != self.prior_generator.num_levels:
            raise ValueError(
                "Num of level_preds is not equal to prior_generator's num_levels."
            )

        level_loc_preds = [x.float() for x in level_loc_preds]
        level_lmk5pt_preds = [x.float() for x in level_lmk5pt_preds]
        level_obj_preds = [x.float() for x in level_obj_preds]

        # batch list targets to level list targets
        level_reg_targets = self._get_targets(
            img_size=img_size,
            loc_gts_list=loc_gts_list,
            lmk5pt_gts_list=lmk5pt_gts_list,
            has_lmk5pt_gts_list=has_lmk5pt_gts_list,
        )

        if level_reg_targets is not None:
            (
                level_flat_priors,
                level_loc_targets,
                level_lmk5pt_targets,
                level_obj_targets,
                _,  # level_loc_weights,
                level_lmk5pt_weights,
                level_obj_weights,
                num_total_pos,
                _,
            ) = level_reg_targets
            device = level_loc_preds[0].device
            num_total_samples = reduce_mean(
                torch.tensor(num_total_pos, dtype=torch.float, device=device)
            )
            num_total_samples = max(num_total_samples.item(), 1.0)

            lbox, lobj, llmk5pt, avg_factor = multi_apply(
                self._calc_level_loss,
                level_loc_preds,
                level_obj_preds,
                level_lmk5pt_preds,
                level_loc_targets,
                level_lmk5pt_targets,
                level_obj_targets,
                # level_loc_weights,
                level_lmk5pt_weights,
                level_obj_weights,
                level_flat_priors,
                self.prior_generator.strides,
                num_total_samples=num_total_samples,
            )
            avg_factor = reduce_mean(sum(avg_factor))
            avg_factor = max(avg_factor.item(), 1.0)
            lbox = sum([x / avg_factor for x in lbox])
            llmk5pt = sum([x / avg_factor for x in llmk5pt])
            lobj = sum(lobj)
            num_total_samples = torch.tensor(num_total_samples).type_as(lobj)
        else:
            lbox = sum([x.sum() * 0 for x in level_loc_preds])
            llmk5pt = sum([x.sum() * 0 for x in level_lmk5pt_preds])
            lobj = sum([x.sum() * 0 for x in level_obj_preds])
            num_total_samples = torch.zeros_like(lobj)

        out = {
            "lbox": lbox,
            "lobj": lobj,
            "llmk5pt": llmk5pt,
            "loss": sum([lbox, llmk5pt, lobj]),
            "nt": num_total_samples,
        }
        return out

    def _calc_level_loss(
        self,
        loc_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        lmk5pt_preds: torch.Tensor,
        loc_targets: torch.Tensor,
        lmk5pt_targets: torch.Tensor,
        obj_targets: torch.Tensor,
        # loc_weights: torch.Tensor,
        lmk5pt_weights: torch.Tensor,
        obj_weights: torch.Tensor,
        flat_priors: torch.Tensor,
        stride: Tuple[int, int],
        num_total_samples: int,
    ):
        if stride[0] != stride[1]:
            raise ValueError("h stride is not equal to w stride!")

        loc_preds = loc_preds.permute(0, 2, 3, 1).reshape(-1, 4)
        obj_preds = obj_preds.permute(0, 2, 3, 1).reshape(-1, 1)
        lmk5pt_preds = lmk5pt_preds.permute(0, 2, 3, 1).reshape(-1, 10)
        loc_targets = loc_targets.reshape(-1, 4)
        lmk5pt_targets = lmk5pt_targets.reshape(-1, 10)
        lmk5pt_weights = lmk5pt_weights.reshape(-1, 10)
        obj_targets = obj_targets.reshape(-1)
        obj_weights = obj_weights.reshape(-1)

        flat_priors = flat_priors.reshape(-1, 4).type_as(loc_preds)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = 1
        pos_inds = (
            ((obj_targets >= 0) & (obj_targets < bg_class_ind)).nonzero().squeeze(1)
        )
        obj_scores = torch.zeros_like(obj_weights)

        if len(pos_inds):
            level_stride = stride[0]
            pos_prior_centers = get_prior_center(flat_priors[pos_inds]) / level_stride
            pos_loc_level_targets = loc_targets[pos_inds] / level_stride
            pos_loc_level_preds = distance2bbox(pos_prior_centers, loc_preds[pos_inds])

            pos_loc_weights = obj_preds.detach().sigmoid()
            pos_loc_weights = pos_loc_weights.flatten()[pos_inds]

            pos_lmk5pt_level_targets = lmk5pt_targets[pos_inds] / level_stride
            pos_lmk5pt_distance_targets = kps2distance(
                pos_prior_centers, pos_lmk5pt_level_targets
            )
            pos_lmk5pt_distance_preds = lmk5pt_preds[pos_inds]
            pos_lmk5pt_weights = (
                lmk5pt_weights[pos_inds].max(dim=1)[0] * pos_loc_weights
            )
            pos_lmk5pt_weights = pos_lmk5pt_weights.reshape(-1, 1)

            obj_scores[pos_inds] = bbox_overlaps(
                pos_loc_level_preds.detach(),
                pos_loc_level_targets,
                is_aligned=True,
            )

            lloc = self.loc_loss(
                pos_loc_level_preds,
                pos_loc_level_targets,
                weight=pos_loc_weights,
                avg_factor=1.0,
            )
            llmk5pt = self.lmk5pt_loss(
                pos_lmk5pt_distance_preds,
                pos_lmk5pt_distance_targets,
                weight=pos_lmk5pt_weights,
                avg_factor=1.0,
            )
        else:
            lloc = loc_preds.sum() * 0
            llmk5pt = lmk5pt_preds.sum() * 0
            pos_loc_weights = torch.zeros_like(lloc)

        lobj = self.obj_loss(
            obj_preds,
            (obj_targets, obj_scores),
            weight=obj_weights,
            avg_factor=num_total_samples,
        )

        return lloc, lobj, llmk5pt, pos_loc_weights.sum()

    def _get_targets(
        self,
        *,
        img_size: Tuple[int, int],
        loc_gts_list: List[torch.Tensor],
        lmk5pt_gts_list: List[torch.Tensor],
        has_lmk5pt_gts_list: List[torch.Tensor],
    ) -> Optional[tuple]:
        device = loc_gts_list[0].device
        feat_sizes = [
            (int(img_size[0] / stride[0]), int(img_size[1] / stride[1]))
            for stride in self.prior_generator.strides
        ]
        priors, valid_flags = self.get_priors(feat_sizes, img_size, device=device)
        num_level_priors = [len(x) for x in priors]
        flat_priors = torch.cat(priors)
        flat_valid_flags = torch.cat(valid_flags)
        inside_flags = anchor_inside_flags(
            flat_priors, flat_valid_flags, img_size, allowed_border=-1
        )

        targets = multi_apply(
            self._get_one_img_target,
            [flat_priors] * len(loc_gts_list),
            [num_level_priors] * len(loc_gts_list),
            [inside_flags] * len(loc_gts_list),
            loc_gts_list,
            lmk5pt_gts_list,
            has_lmk5pt_gts_list,
        )
        out = None

        if targets is not None:
            (
                flat_priors_list,
                loc_targets_list,
                lmk5pt_targets_list,
                obj_targets_list,
                loc_weights_list,
                lmk5pt_weights_list,
                obj_weights_list,
                pos_inds_list,
                neg_inds_list,
            ) = targets
            # valid priors
            num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
            num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
            # split targets to a list w.r.t. multiple levels
            level_flat_priors = images_to_levels(flat_priors_list, num_level_priors)
            level_loc_targets = images_to_levels(loc_targets_list, num_level_priors)
            level_lmk5pt_targets = images_to_levels(
                lmk5pt_targets_list, num_level_priors
            )
            level_obj_targets = images_to_levels(obj_targets_list, num_level_priors)
            level_loc_weights = images_to_levels(loc_weights_list, num_level_priors)
            level_lmk5pt_weights = images_to_levels(
                lmk5pt_weights_list, num_level_priors
            )
            level_obj_weights = images_to_levels(obj_weights_list, num_level_priors)

            out = (
                level_flat_priors,
                level_loc_targets,
                level_lmk5pt_targets,
                level_obj_targets,
                level_loc_weights,
                level_lmk5pt_weights,
                level_obj_weights,
                num_total_pos,
                num_total_neg,
            )

        return out

    def _get_one_img_target(
        self,
        flat_priors: torch.Tensor,
        num_level_priors: List[int],
        inside_flags: torch.Tensor,
        loc_gts: torch.Tensor,
        lmk5pt_gts: torch.Tensor,
        has_lmk5pt_gts: torch.Tensor,
    ) -> tuple:
        if inside_flags.any():
            # assign gt and sample priors
            inside_priors = flat_priors[inside_flags]
            pred_instances = InstanceData(metainfo={"priors": inside_priors})
            labels = torch.ones(len(loc_gts), dtype=torch.long, device=loc_gts.device)
            gt_instances = InstanceData(metainfo={"bboxes": loc_gts, "labels": labels})
            num_level_priors_inside = self._get_num_level_priors_inside(
                num_level_priors, inside_flags
            )
            assign_result = self.assigner.assign(
                pred_instances=pred_instances,
                num_level_priors=num_level_priors_inside,
                gt_instances=gt_instances,
            )
            sampling_result = self.sampler.sample(
                assign_result, pred_instances, gt_instances
            )

            num_inside_priors = inside_priors.shape[0]
            loc_targets = torch.zeros_like(inside_priors)
            loc_weights = torch.zeros_like(inside_priors)
            lmk5pt_targets = inside_priors.new_zeros(size=(num_inside_priors, 10))
            lmk5pt_weights = inside_priors.new_zeros(size=(num_inside_priors, 10))
            obj_targets = inside_priors.new_full(
                size=(num_inside_priors,), fill_value=-1, dtype=torch.long
            )
            obj_weights = inside_priors.new_ones(size=(num_inside_priors,))

            # pos_anchors = sampling_result.pos_priors
            # assign_pos_inds = sampling_result.pos_assigned_gt_inds
            # pos_gt_anchor_ious = bbox_overlaps(loc_gts[assign_pos_inds], pos_anchors, is_aligned=True)
            neg_inds = sampling_result.neg_inds
            pos_inds = sampling_result.pos_inds
            # is_pos_mask = pos_gt_anchor_ious >= 0
            # neg_inds = torch.concat((neg_inds, pos_inds[~is_pos_mask]), dim=0)
            # pos_inds = pos_inds[is_pos_mask]
            # assign_pos_inds = assign_pos_inds[is_pos_mask]
            if len(pos_inds):
                loc_targets[pos_inds] = sampling_result.pos_gt_bboxes
                loc_weights[pos_inds] = 1.0

                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
                lmk5pt_targets[pos_inds] = lmk5pt_gts[pos_assigned_gt_inds]
                lmk5pt_weights[pos_inds] = (
                    has_lmk5pt_gts[pos_assigned_gt_inds].unsqueeze(1).repeat(1, 10)
                )

                obj_targets[pos_inds] = 0

            # map up to original set of priors
            num_total_priors = len(flat_priors)
            flat_priors = unmap(flat_priors, num_total_priors, inside_flags)
            loc_targets = unmap(loc_targets, num_total_priors, inside_flags)
            loc_weights = unmap(loc_weights, num_total_priors, inside_flags)
            lmk5pt_targets = unmap(lmk5pt_targets, num_total_priors, inside_flags)
            lmk5pt_weights = unmap(lmk5pt_weights, num_total_priors, inside_flags)
            obj_targets = unmap(obj_targets, num_total_priors, inside_flags, fill=1)
            obj_weights = unmap(obj_weights, num_total_priors, inside_flags)

            out = (
                flat_priors,
                loc_targets,
                lmk5pt_targets,
                obj_targets,
                loc_weights,
                lmk5pt_weights,
                obj_weights,
                pos_inds,
                neg_inds,
            )

            return out

    def _get_num_level_priors_inside(self, num_level_priors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_priors)
        num_level_priors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_priors_inside

    def _get_one_assign_priors(
        self,
        img_size: Tuple[int, int],
        loc_gts: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = loc_gts.device
        feat_sizes = [
            (int(img_size[0] / stride[0]), int(img_size[1] / stride[1]))
            for stride in self.prior_generator.strides
        ]
        priors, valid_flags = self.get_priors(feat_sizes, img_size, device=device)
        num_level_priors = [len(x) for x in priors]
        flat_priors = torch.cat(priors)
        flat_valid_flags = torch.cat(valid_flags)
        inside_flags = anchor_inside_flags(
            flat_priors, flat_valid_flags, img_size, allowed_border=-1
        )
        lmk5pt_gts = torch.full((len(loc_gts), 10), -1).type_as(loc_gts)
        has_lmk5pt_gts = torch.full((len(loc_gts),), -1).type_as(loc_gts)

        outs = self._get_one_img_target(
            flat_priors=flat_priors,
            num_level_priors=num_level_priors,
            inside_flags=inside_flags,
            loc_gts=loc_gts,
            lmk5pt_gts=lmk5pt_gts,
            has_lmk5pt_gts=has_lmk5pt_gts,
        )
        flat_priors, _, _, bg_targets, *_, num_level_priors = outs
        level_priors = [x[0] for x in images_to_levels([flat_priors], num_level_priors)]
        level_bg = [x[0] for x in images_to_levels([bg_targets], num_level_priors)]
        level_assign_priors = [
            prior[bg == 0] for prior, bg in zip(level_priors, level_bg)
        ]
        return level_priors, level_assign_priors

    @staticmethod
    def _gen_raw_score_map(score_map, size):
        score_map_size = score_map.shape[:2]
        size = np.array(size)
        scale = size / score_map_size
        shift = (scale // 2).astype(int).tolist()
        out_score_map = cb.imresize(score_map, tuple(size.tolist()))
        out_score_map = cv2.copyMakeBorder(
            out_score_map, 0, shift[0], 0, shift[1], cv2.BORDER_REPLICATE
        )
        return out_score_map[shift[1]:, shift[0]:]

    def plot_assign_priors(
        self,
        img: np.ndarray,
        loc_gts: torch.Tensor,
        score_maps: np.ndarray = None,
        draw_prior_box: bool = False,
    ):
        img_h, img_w = img.shape[:2]
        level_priors, level_assign_priors = self._get_one_assign_priors(
            img_size=(img_h, img_w),
            loc_gts=loc_gts,
        )
        level_plots = []
        score_maps = [None] * len(level_priors) if score_maps is None else score_maps

        base_sizes = self.prior_generator.base_sizes
        scales = self.prior_generator.scales.tolist()

        zipped = zip(level_priors, level_assign_priors, score_maps, base_sizes)
        for i, (priors, assign_priors, score_map, base_size) in enumerate(zipped):
            plotted = img.copy().astype("uint8")
            prior_centers = [
                tuple(x) for x in get_prior_center(priors).cpu().int().tolist()
            ]
            assign_priors = [tuple(x) for x in assign_priors.cpu().int().tolist()]
            score_map = [None] * 2 if score_map is None else score_map
            plotted_outs = []

            for map_, scale in zip(score_map, scales):
                tmp_plotted = plotted.copy()

                if map_ is not None:
                    map_ = SCRFDHead._gen_raw_score_map(map_, tmp_plotted.shape[:2])
                    heatmap = cv2.applyColorMap(map_, cv2.COLORMAP_JET)
                    tmp_plotted = cv2.addWeighted(tmp_plotted, 0.3, heatmap, 0.7, 0.1)

                tmp_plotted = cb.draw_points(
                    tmp_plotted,
                    prior_centers,
                    scales=i * 0.2 + 0.2,
                    colors=(85, 85, 85),
                )

                if len(assign_priors):
                    prior_heights = cb.Boxes(assign_priors).height
                    prior_mask_to_plot = prior_heights / base_size == scale
                    for is_plot, prior_box in zip(prior_mask_to_plot, assign_priors):
                        if is_plot:
                            prior_box = cb.Box(prior_box)
                            prior_center = tuple(
                                prior_box.center.round().astype(int).tolist()
                            )
                            tmp_plotted = cb.draw_point(tmp_plotted, prior_center, scale=i * 0.3 + 0.5, color=(255, 255, 255))
                            if draw_prior_box:
                                tmp_plotted = cb.draw_box(tmp_plotted, prior_box, color=(255, 255, 255), thickness=1)

                plotted_outs.append(tmp_plotted)

            plotted_outs = np.concatenate(plotted_outs, axis=0)
            level_plots.append(plotted_outs)
        level_plots = np.concatenate(level_plots, axis=1)
        return level_plots
