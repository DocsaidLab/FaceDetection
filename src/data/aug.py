import math
import random
from typing import List, Tuple, Union

import capybara as cb
import cv2
import numpy as np
from capybara.vision.visualization.utils import _Color, prepare_color
from chameleon import Registry

AUGMENTATIONS = Registry("augmentations")


@AUGMENTATIONS.register_module()
class RandomHFlip:
    def __init__(self, prob) -> None:
        self.prob = prob

    @staticmethod
    def _box_flip(bboxes, img_shape):
        flipped = bboxes.copy()
        w = img_shape[1]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
        return flipped

    @staticmethod
    def _kps_flip(kps, img_shape):
        flipped = kps.copy()
        flip_order = [1, 0, 2, 4, 3]
        for idx, a in enumerate(flip_order):
            flipped[:, idx, :] = kps[:, a, :]
        w = img_shape[1]
        flipped[..., 0] = w - flipped[..., 0]
        return flipped

    @staticmethod
    def _pose_flip(poses):
        pose_dict = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 5, 6: 6, -1: -1}
        return np.array([pose_dict[i] for i in poses.tolist()])

    def __call__(self, data):
        if self.prob > np.random.uniform(0, 1):
            img_shape = data["image"].shape
            data["image"] = cv2.flip(data["image"], 1)
            data["boxes"] = self._box_flip(data["boxes"], img_shape)
            data["lmk5pts"] = self._kps_flip(data["lmk5pts"], img_shape)
            # data['poses'] = self._pose_flip(data['poses'])
        return data


@AUGMENTATIONS.register_module()
class PhotoMetricDistortion:
    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18,
        gray_prob: float = 0.0,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.gray_prob = gray_prob

    def __call__(self, data):
        img = (
            np.asarray(data["image"], dtype="float32")
            if data["image"].dtype != np.float32
            else data["image"]
        )

        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(
                -self.brightness_delta,
                self.brightness_delta,
            )
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(
                    self.contrast_lower,
                    self.contrast_upper,
                )
                img *= alpha

        # convert color from BGR to HSV
        img = cb.imcvtcolor(img, "BGR2HSV")

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(
                self.saturation_lower, self.saturation_upper
            )

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = cb.imcvtcolor(img, "HSV2BGR")

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]

        if np.random.random() < self.gray_prob:
            gray = cb.imcvtcolor(img, "BGR2GRAY")
            img = cv2.merge([gray, gray, gray])

        data["image"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@AUGMENTATIONS.register_module()
class Resize:
    def __init__(
        self,
        h: int,
        w: int,
        interpolation: Union[str, int, cb.INTER] = cb.INTER.BILINEAR,
        enable_random_interpolation: bool = False,
    ):
        self.h = h
        self.w = w
        self.interpolation = interpolation
        self.enable_random_interpolation = enable_random_interpolation

    def __call__(self, data):
        raw_h, raw_w = data["image"].shape[:2]
        if self.enable_random_interpolation:
            interpolation = cb.INTER(random.randint(0, 4))
        else:
            interpolation = self.interpolation
        data["image"] = cb.imresize(
            data["image"],
            (self.h, self.w),
            return_scale=False,
            interpolation=interpolation,
        )
        new_h, new_w = data["image"].shape[:2]
        w_scale = new_w / raw_w
        h_scale = new_h / raw_h
        data["boxes"] *= (w_scale, h_scale, w_scale, h_scale)
        data["lmk5pts"] *= (w_scale, h_scale)
        data["interpolation"] = interpolation
        return data


@AUGMENTATIONS.register_module()
class ResizePadIfNeed:
    def __init__(
        self,
        h: int,
        w: int,
        interpolation: Union[str, int, cb.INTER] = cb.INTER.BILINEAR,
        enable_random_interpolation: bool = False,
        pad_val: _Color = 0,
        pad_mode: str = "CONSTANT",
    ):
        self.h = h
        self.w = w
        self.interpolation = interpolation
        self.enable_random_interpolation = enable_random_interpolation
        self.pad_val = prepare_color(pad_val)
        self.pad_mode = pad_mode

    def __call__(self, data):
        dst_size = data.get("dst_size", None)

        dst_h, dst_w = (self.h, self.w) if dst_size is None else dst_size

        img = data["image"]

        raw_h, raw_w = img.shape[:2]
        scale = min(dst_h / raw_h, dst_w / raw_w)

        tmp_h, tmp_w = (
            min(dst_h, math.ceil(raw_h * scale)),
            min(dst_w, math.ceil(raw_w * scale)),
        )

        if self.enable_random_interpolation:
            interpolation = cb.INTER(random.randint(0, 4))
        else:
            interpolation = self.interpolation

        img, w_scale, h_scale = cb.imresize(
            img,
            size=(tmp_h, tmp_w),
            return_scale=True,
            interpolation=interpolation,
        )

        img_h, img_w = img.shape[:2]
        delta_h, delta_w = dst_h - img_h, dst_w - img_w
        dst_img = cb.pad(
            img,
            pad_size=(0, delta_h, 0, delta_w),
            pad_value=self.pad_val,
            pad_mode=self.pad_mode,
        )

        data["image"] = dst_img
        data["interpolation"] = interpolation

        boxes = data.get("boxes", None)
        if boxes is not None:
            data["boxes"] *= (w_scale, h_scale, w_scale, h_scale)

        lmk5pts = data.get("lmk5pts", None)
        if lmk5pts is not None:
            data["lmk5pts"] *= (w_scale, h_scale)
        return data


def _is_point_in_patch_and_size_ok(points, patch):
    # TODO >=
    mask = (
        (points[:, 0] > patch[0])
        * (points[:, 1] > patch[1])
        * (points[:, 0] < patch[2])
        * (points[:, 1] < patch[3])
    )
    return mask


def _is_bboxes_has_min_side_ratio_in_patch_and_size_ok(boxes, patch, min_side_ratio):
    # TODO >=
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    delta_ws, delta_hs = (min_side_ratio - 0.5) * ws, (min_side_ratio - 0.5) * hs
    delta = np.stack((delta_ws, delta_hs), -1)

    four_points = np.stack(
        [
            centers - delta,
            centers - (1, -1) * delta,
            centers - (-1, 1) * delta,
            centers + delta,
        ]
    )
    mask = np.stack(
        [_is_point_in_patch_and_size_ok(points, patch) for points in four_points]
    )
    mask = np.all(mask, 0)
    return mask


def _is_center_of_bboxes_in_patch(boxes, patch):
    # TODO >=
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    mask = (
        (center[:, 0] > patch[0])
        * (center[:, 1] > patch[1])
        * (center[:, 0] < patch[2])
        * (center[:, 1] < patch[3])
    )
    return mask


# @AUGMENTATIONS.register_module()
# class RandomSquareCropSCRFD:

#     def __init__(
#         self,
#         crop_choice: List[float] = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
#         pad_val: _Color = None,
#     ):
#         self.crop_choice = crop_choice
#         self.pad_val = prepare_color(pad_val)

#     def __call__(self, data):
#         img = data['image'].copy()
#         boxes = data['boxes'].copy()
#         lmk5pts = data['lmk5pts'].copy()
#         has_lmk5pts = data['has_lmk5pts'].copy()
#         h, w, _ = img.shape

#         max_scale = np.amax(self.crop_choice)
#         pad_val = tuple(np.random.randint(0, 255, size=3).tolist()) if self.pad_val is None else self.pad_val
#         pad_val = prepare_color(pad_val)
#         scale_retry = 0
#         while True:
#             scale_retry += 1

#             if scale_retry == 1 or max_scale > 1.0:
#                 scale = np.random.choice(self.crop_choice)
#             else:
#                 scale = scale * 1.2

#             for _ in range(250):
#                 short_side = min(w, h)
#                 ch = cw = int(scale * short_side)

#                 if w == cw:
#                     left = 0
#                 elif w > cw:
#                     left = np.random.randint(0, w - cw)
#                 else:
#                     left = np.random.randint(w - cw, 0)
#                 if h == ch:
#                     top = 0
#                 elif h > ch:
#                     top = np.random.randint(0, h - ch)
#                 else:
#                     top = np.random.randint(h - ch, 0)

#                 patch = np.array((left, top, left + cw, top + ch), dtype='int')

#                 # center of boxes should inside the crop img
#                 # only adjust boxes and instance masks when the gt is not empty
#                 # adjust boxes

#                 # mask = _is_bboxes_has_min_side_ratio_in_patch_and_size_ok(boxes, patch, 0.4)
#                 mask = _is_center_of_bboxes_in_patch(boxes, patch)
#                 if not mask.any():
#                     continue

#                 boxes = boxes[mask]
#                 boxes[:, :4] = boxes[:, :4] - np.tile(patch[:2], 2)
#                 data['boxes'] = boxes

#                 lmk5pts = lmk5pts[mask]
#                 lmk5pts[..., 0] -= patch[0]
#                 lmk5pts[..., 1] -= patch[1]
#                 data['lmk5pts'] = lmk5pts
#                 data['has_lmk5pts'] = has_lmk5pts[mask]
#                 # data['poses'] = poses[mask]

#                 # adjust the img no matter whether the gt is empty before crop
#                 rimg = np.full((ch, cw, 3), fill_value=pad_val, dtype=img.dtype)
#                 patch_from = patch.copy()
#                 patch_from[0] = max(0, patch_from[0])
#                 patch_from[1] = max(0, patch_from[1])
#                 patch_from[2] = min(img.shape[1], patch_from[2])
#                 patch_from[3] = min(img.shape[0], patch_from[3])
#                 patch_to = patch.copy()
#                 patch_to[0] = max(0, patch_to[0] * -1)
#                 patch_to[1] = max(0, patch_to[1] * -1)
#                 patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
#                 patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])

#                 rimg[patch_to[1]:patch_to[3], patch_to[0]:patch_to[2]] = \
#                     img[patch_from[1]:patch_from[3], patch_from[0]:patch_from[2]]
#                 data['image'] = rimg
#                 return data


@AUGMENTATIONS.register_module()
class RandomSquareCrop:
    def __init__(
        self,
        crop_choice: List[float] = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        pad_val: _Color = 0,
    ):
        self.crop_choice = crop_choice
        self.pad_val = prepare_color(pad_val)

    def __call__(self, data):
        img = data["image"].copy()
        boxes = data["boxes"].copy()
        lmk5pts = data["lmk5pts"].copy()
        has_lmk5pts = data["has_lmk5pts"].copy()
        h, w, _ = img.shape

        max_scale = np.amax(self.crop_choice)
        pad_val = (
            tuple(np.random.randint(0, 255, size=3).tolist())
            if self.pad_val is None
            else self.pad_val
        )
        pad_val = prepare_color(pad_val)
        scale_retry = 0
        while True:
            scale_retry += 1

            if scale_retry == 1 or max_scale > 1.0:
                scale = np.random.choice(self.crop_choice)
            else:
                scale = scale * 1.2

            for _ in range(250):
                short_side = min(w, h)
                ch = cw = int(scale * short_side)

                if w == cw:
                    left = 0
                elif w > cw:
                    left = np.random.randint(0, w - cw)
                else:
                    left = np.random.randint(w - cw, 0)
                if h == ch:
                    top = 0
                elif h > ch:
                    top = np.random.randint(0, h - ch)
                else:
                    top = np.random.randint(h - ch, 0)

                patch = np.array((left, top, left + cw, top + ch), dtype="int")

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                # adjust boxes

                # mask = _is_bboxes_has_min_side_ratio_in_patch_and_size_ok(boxes, patch, 0.4)
                mask = _is_center_of_bboxes_in_patch(boxes, patch)
                if not mask.any():
                    continue

                boxes = boxes[mask]
                boxes[:, :4] = boxes[:, :4] - np.tile(patch[:2], 2)
                data["boxes"] = boxes

                lmk5pts = lmk5pts[mask]
                lmk5pts[..., 0] -= patch[0]
                lmk5pts[..., 1] -= patch[1]
                data["lmk5pts"] = lmk5pts
                data["has_lmk5pts"] = has_lmk5pts[mask]
                # data['poses'] = poses[mask]

                # adjust the img no matter whether the gt is empty before crop
                patch_from = patch.copy()
                patch_from[0] = max(0, patch_from[0])
                patch_from[1] = max(0, patch_from[1])
                patch_from[2] = min(img.shape[1], patch_from[2])
                patch_from[3] = min(img.shape[0], patch_from[3])
                patch_to = patch.copy()
                patch_to[0] = max(0, patch_to[0] * -1)
                patch_to[1] = max(0, patch_to[1] * -1)
                patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
                patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])

                cropped_img = img[
                    patch_from[1] : patch_from[3], patch_from[0] : patch_from[2]
                ]
                left, right, top, bottom = (
                    patch_to[0],
                    max(0, cw - patch_to[2]),
                    patch_to[1],
                    max(0, ch - patch_to[3]),
                )
                rimg = cb.pad(
                    cropped_img, pad_size=(top, bottom, left, right), pad_value=pad_val
                )
                data["image"] = rimg
                return data


@AUGMENTATIONS.register_module()
class AnnsClipper:
    def __call__(self, data):
        img_h, img_w = data["image"].shape[:2]
        data["boxes"][:, 0::2] = np.clip(data["boxes"][:, 0::2], 0, img_w)
        data["boxes"][:, 1::2] = np.clip(data["boxes"][:, 1::2], 0, img_h)
        if "lmk5pts" in data:
            data["lmk5pts"][..., 0] = np.clip(data["lmk5pts"][..., 0], 0, img_w)
            data["lmk5pts"][..., 1] = np.clip(data["lmk5pts"][..., 1], 0, img_h)
        return data

    def __repr__(self):
        return self.__class__.__name__


@AUGMENTATIONS.register_module()
class Normalize:
    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype="float32")
        self.std = np.array(std, dtype="float32")
        self.to_rgb = to_rgb

    def __call__(self, data):
        data["image"] = (
            cb.imcvtcolor(data["image"], "BGR2RGB") if self.to_rgb else data["image"]
        )
        data["image"] = (data["image"] - self.mean) / self.std
        data["mean"] = self.mean
        data["std"] = self.std
        data["color_base"] = "rgb" if self.to_rgb else "bgr"
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std})"
        return repr_str


@AUGMENTATIONS.register_module()
class ToFloat32:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            data[k] = np.ascontiguousarray(data[k], dtype="float32")
        return data


class Transforms:
    def __init__(self, transforms: List[dict] = []):
        self.transforms = [AUGMENTATIONS.build(cfg) for cfg in transforms]

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
