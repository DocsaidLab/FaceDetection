import json
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import capybara as cb
import numpy as np
import torch
from chameleon import Registry
from scipy.io import loadmat
from torch.utils.data import Dataset

from ..utils import draw_results
from .aug import Transforms
from .utils import dict_to_anns

DATASETS = Registry("datasets")


def load_data(json_fpath) -> Dict[str, np.ndarray]:
    with open(json_fpath, "r") as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)
        img_path = cb.Path(json_fpath).parent / json_data["img_fname"]
        img = cb.imread(img_path)
        anns = dict_to_anns(json_data)
        anns.pop("poses")
    return {"image": img, "img_path": str(img_path), **anns}


def prepare_data_for_train(data) -> Dict[str, torch.Tensor]:
    data["boxes"] = data.get("boxes", np.empty((0, 4), dtype="float32"))
    data["lmk5pts"] = data.get("lmk5pts", np.empty((0, 5, 2), dtype="float32"))
    data["has_lmk5pts"] = data.get("has_lmk5pts", np.empty((0, 1), dtype="int"))
    data["boxes_area"] = np.full((0), fill_value=-1, dtype="int")

    n_faces = len(data["boxes"])
    if n_faces != len(data["lmk5pts"]):
        raise ValueError("number of boxes is not equal to number of keypoints_list")

    if n_faces:
        boxes = data["boxes"]
        has_lmk5pts = data["has_lmk5pts"]
        lmk5pts = data["lmk5pts"]
        data["boxes_area"] = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )
        mask = has_lmk5pts.flatten() == 0
        lmk5pts[mask] = -1
        data["lmk5pts"] = lmk5pts
    return data


@DATASETS.register_module()
class DatasetFromJson(Dataset):
    def __init__(
        self,
        txt_path,
        num_samples: Optional[int] = None,
        transforms: List[dict] = [],
    ):
        super().__init__()
        txt_path = cb.Path(txt_path)
        with open(txt_path, "r") as f:
            json_fpaths = [txt_path.parent / x for x in map(str.strip, f.readlines())]
        self.json_fpaths = (
            json_fpaths[:num_samples] if num_samples is not None else json_fpaths
        )
        self.num_samples = len(self.json_fpaths)
        self.transforms = Transforms(transforms)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        if isinstance(item, (tuple, list)):
            ind, dst_size = item
        else:
            ind = item
            dst_size = None

        if ind > len(self):
            raise IndexError("Given index is out of range.")

        data = load_data(self.json_fpaths[ind])
        if dst_size is not None:
            data["dst_size"] = dst_size
        data = self.transforms(data)
        data = prepare_data_for_train(data)
        return data


@DATASETS.register_module()
class ConcatDatasets(Dataset):
    def __init__(
        self,
        datasets: List[dict] = [],
        dataset_weights: List[float] = None,
        total_samples: int = -1,
        transforms: List[dict] = [],
    ):
        super().__init__()

        datasets = [DATASETS.build(x) for x in datasets]
        self.total_samples = (
            sum([len(dataset) for dataset in datasets])
            if total_samples == -1
            else total_samples
        )

        if dataset_weights is not None:
            if len(datasets) != len(dataset_weights):
                raise ValueError(
                    "Length of datasets and dataset_weights should be the same."
                )
            if not np.isclose(sum(dataset_weights), 1):
                raise ValueError("Sum of dataset_weights should be 1.")
        self.datasets = datasets
        self.dataset_weights = dataset_weights
        self.transforms = Transforms(transforms)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, item: Union[int, Tuple[int, Tuple[int, int]]]) -> Any:
        if isinstance(item, tuple):
            ind, dst_size = item
        else:
            ind = item
            dst_size = None

        if ind >= len(self):
            raise IndexError(f"Index is out of dataset.")

        if self.dataset_weights is not None:
            data_inds = list(range(len(self.datasets)))
            data_ind = np.random.choice(data_inds, p=self.dataset_weights)
            dataset = self.datasets[data_ind]
            ind = np.random.randint(0, len(dataset))
        else:
            total = upper_total = 0
            data_ind = 0
            for data_ind, dataset in enumerate(self.datasets):
                upper_total += len(dataset)
                if upper_total > ind:
                    ind -= total
                    break
                total = upper_total

        item = (ind, dst_size) if isinstance(item, tuple) else ind
        data = dataset[item]
        if dst_size is not None:
            data["dst_size"] = dst_size
        data = self.transforms(data)
        data = prepare_data_for_train(data)
        return data


@DATASETS.register_module()
class WiderFaceEvaluation(Dataset):
    def __init__(
        self,
        data_folder: str,
        gt_folder: str,
        transforms: List[dict] = [],
    ):
        super().__init__()
        data_folder = cb.Path(data_folder)
        gt_folder = cb.Path(gt_folder)
        self.files = cb.get_files(data_folder, suffix=[".jpg", ".jpeg", ".png"])
        self.gt_dict = self._get_gt_boxes(gt_folder)
        self.transforms = Transforms(transforms)

    @staticmethod
    def _get_gt_boxes(gt_folder) -> dict:
        gt_mat = loadmat(str(gt_folder / "wider_face_val.mat"))
        hard_mat = loadmat(str(gt_folder / "wider_hard_val.mat"))
        medium_mat = loadmat(str(gt_folder / "wider_medium_val.mat"))
        easy_mat = loadmat(str(gt_folder / "wider_easy_val.mat"))

        boxes_list = gt_mat["face_bbx_list"]
        event_list = gt_mat["event_list"]
        files_list = gt_mat["file_list"]

        hard_gt_list = hard_mat["gt_list"]
        medium_gt_list = medium_mat["gt_list"]
        easy_gt_list = easy_mat["gt_list"]

        gts_dict = {}
        zipped = zip(
            boxes_list,
            event_list,
            files_list,
            hard_gt_list,
            medium_gt_list,
            easy_gt_list,
        )
        for boxes, event, files, hards, mediums, easys in zipped:
            boxes = [x[0] for x in boxes[0]]
            files = list(*chain([[x[0][0].tolist() for x in xs] for xs in files]))
            event = event[0][0]
            hards = [x[0].flatten() for x in hards[0]]
            mediums = [x[0].flatten() for x in mediums[0]]
            easys = [x[0].flatten() for x in easys[0]]
            for file, b, h, m, e in zip(files, boxes, hards, mediums, easys):
                levels = [-1] * len(b)
                for x in h:
                    levels[x - 1] = 2
                for x in m:
                    levels[x - 1] = 1
                for x in e:
                    levels[x - 1] = 0
                tmp = {
                    f"{file}": {
                        "boxes": b,
                        "levels": levels,
                    }
                }
                gts_dict.update(**tmp)
        return gts_dict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind):
        if ind >= len(self):
            raise IndexError("Given index is out of range.")
        fpath = self.files[ind]
        img = cb.imread(self.files[ind])
        gts = self.gt_dict[fpath.stem]
        boxes = cb.Boxes(np.array(gts["boxes"]), "XYWH").convert("XYXY")
        data = {
            "image": img,
            "boxes": boxes.numpy(),
        }
        data = self.transforms(data)
        data["levels"] = np.array(gts["levels"], dtype="float32")
        return data


def plot_data(image, boxes, lmk5pts, mean=None, std=None, color_base="bgr", **kwargs):
    image = image.numpy() if isinstance(image, torch.Tensor) else image
    std = std.numpy() if isinstance(std, torch.Tensor) else std
    mean = mean.numpy() if isinstance(mean, torch.Tensor) else mean
    boxes = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    lmk5pts = (
        lmk5pts.numpy().reshape(-1, 5, 2)
        if isinstance(lmk5pts, torch.Tensor)
        else lmk5pts
    )

    image = image * std + mean
    image = image.astype("uint8").transpose(1, 2, 0)
    image = (
        cb.imcvtcolor(image, f"{color_base.upper()}2BGR")
        if color_base != "bgr"
        else image
    )
    plotted = draw_results(
        image,
        boxes=cb.Boxes(boxes),
        kpts_list=cb.KeypointsList(lmk5pts),
        show_score_bar=False,
    )
    return plotted


def plot_batch_data(batch):
    plotteds = [
        plot_data(image, boxes, lmk5pts, mean, std, color_base)
        for image, boxes, lmk5pts, mean, std, color_base in zip(
            batch["image"],
            batch["boxes"],
            batch["lmk5pts"],
            batch.get("mean"),
            batch.get("std"),
            batch.get("color_base"),
        )
    ]
    return plotteds


def collate_fn(batch):
    outs = {}
    for k, v in batch[0].items():
        if k == "image":
            outs[k] = (
                torch.from_numpy(np.stack([x["image"] for x in batch]))
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
            )
        elif k in ["boxes", "lmk5pts", "has_lmk5pts", "boxes_area"]:
            outs[k] = [torch.from_numpy(x[k]).contiguous().float() for x in batch]
        else:
            if isinstance(v, (int, float, np.ndarray)):
                outs[k] = (
                    torch.from_numpy(np.stack([x[k] for x in batch]))
                    .contiguous()
                    .float()
                )
            else:
                outs[k] = [x[k] for x in batch]
    outs["boxes"] = [x.reshape(-1, 4) for x in outs["boxes"]]
    outs["lmk5pts"] = [x.reshape(-1, 10) for x in outs["lmk5pts"]]
    outs["has_lmk5pts"] = [x.reshape(-1) for x in outs["has_lmk5pts"]]
    return outs
