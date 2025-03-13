import random
from typing import List

import albumentations as A
import capybara as cb
import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from src.data.enums import FacePose
from src.data.utils import dict_to_anns, oneface_to_dict


def prepare_data(data, pose_group, raw_min_face_width):
    ind = np.random.randint(0, len(data))
    tmp_data = data[ind]
    img = cb.imread(tmp_data["img_fpath"])
    anns = dict_to_anns(tmp_data)

    poses = [FacePose(x.item()) for x in anns["poses"].flatten().astype("int")]
    inds = []
    for i, pose in enumerate(poses):
        box = cb.Box(anns["boxes"][i])
        if (
            pose in pose_group
            and box.width >= raw_min_face_width
            and anns["has_lmk5pts"][i]
        ):
            inds.append(i)

    sample = {}
    if len(inds):
        ind = np.random.choice(inds).tolist()
        box = anns["boxes"][ind]
        lmk5pt = anns["lmk5pts"][ind]
        pose = poses[ind]
        sample = {
            "img": img,
            "box": box,
            "lmk5pt": lmk5pt,
            "pose": pose,
            "img_fpath": tmp_data["img_fpath"],
        }
    return sample


affine = A.Compose(
    [A.Affine(scale=1, rotate=[-10, 10], shear=[-5, 5], rotate_method="ellipse", p=1)],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], clip=True),
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    p=1,
)

blur = A.OneOf(
    [
        A.Blur((3, 7)),
        A.GaussianBlur((3, 7)),
        A.RingingOvershoot(),
        A.ShotNoise((0.01, 0.03)),
        A.MotionBlur(5),
        A.ZoomBlur((1, 1.1)),
        A.RandomSunFlare(src_radius=50),
    ],
    p=1,
)


def get_dst_face_width(
    mean_face_width: int,
    std_face_width: int,
    max_face_width: int,
    min_face_width: int,
):
    i = 0
    while i < 100:
        dst_face_width = np.random.normal(mean_face_width, std_face_width)
        i += 1
        if min_face_width <= dst_face_width <= max_face_width:
            break
    if i == 100:
        dst_face_width = np.clip(dst_face_width, min_face_width, max_face_width)
    return dst_face_width


def aug_data(
    data: dict,
    mean_face_width: int,
    std_face_width: int,
    max_face_width: int,
    min_face_width: int,
    out_h: int = 300,
    out_w: int = 252,
    distortion_p: float = 0,
    blur_p: float = 0,
):
    tmp = {
        "image": data["img"],
        "bboxes": [data["box"]],
        "keypoints": data["lmk5pt"],
        "labels": [
            1,
        ],
    }
    while True:
        out = affine(**tmp) if random.random() < distortion_p else tmp
        if len(out["bboxes"]):
            break
    out = blur(**out) if random.random() < blur_p else out

    data["img"] = out["image"]
    data["box"] = np.array(out["bboxes"][0])
    data["lmk5pt"] = np.array(out["keypoints"])

    dst_face_width = get_dst_face_width(
        mean_face_width, std_face_width, max_face_width, min_face_width
    )
    tmp_scale = dst_face_width / cb.Box(data["box"]).width
    raw_h, raw_w = data["img"].shape[:2]
    data["img"] = cv2.resize(data["img"], dsize=None, fx=tmp_scale, fy=tmp_scale)
    dst_h, dst_w = data["img"].shape[:2]
    scale_h, scale_w = (dst_h / raw_h, dst_w / raw_w)

    box = cb.Box(data["box"] * ((scale_w, scale_h) * 2))
    lmk5pt = cb.Keypoints(data["lmk5pt"] * (scale_w, scale_h))

    cx, cy = box.center
    face_w, face_h = box.width, box.height
    crop_box = cb.Box((int(cx), int(cy), out_w, out_h), "CXCYWH")
    img = cb.imcropbox(data["img"], crop_box, use_pad=True)

    shift_x, shift_y = crop_box.left_top
    face_box = cb.Box((int(cx), int(cy), face_w, face_h), "CXCYWH").convert("XYXY")
    face_box = face_box.shift(shift_x=-shift_x, shift_y=-shift_y)
    lmk5pt = lmk5pt.shift(shift_x=-shift_x, shift_y=-shift_y)
    data["img"] = img
    data["box"] = face_box.numpy()
    data["lmk5pt"] = lmk5pt.numpy()
    return data


def make_data(
    data: List[dict],
    n_samples: int,
    raw_min_face_width: int,
    out_h: int,
    out_w: int,
    mean_face_width: int,
    std_face_width: int,
    max_face_width: int,
    min_face_width: int,
    pose_group: List[FacePose] = [FacePose.Frontal],
    distortion_p: float = 0,
    blur_p: float = 0,
):
    samples = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        TextColumn(
            "[progress.percentage]{task.completed}/{task.total} ({task.percentage:.2f}%)"
        ),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[yellow]Making data...", total=n_samples)
        while not progress.finished:
            sample = prepare_data(data, pose_group, raw_min_face_width)
            if len(sample):
                sample = aug_data(
                    sample,
                    mean_face_width=mean_face_width,
                    std_face_width=std_face_width,
                    max_face_width=max_face_width,
                    min_face_width=min_face_width,
                    out_h=out_h,
                    out_w=out_w,
                    distortion_p=distortion_p,
                    blur_p=blur_p,
                )
                anns = oneface_to_dict(sample["box"], sample["lmk5pt"], sample["pose"])
                out = {
                    "img": sample["img"],
                    "anns": anns,
                }
                samples.append(out)
                progress.update(task, advance=1)
    return samples
