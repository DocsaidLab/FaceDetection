import json
from typing import Dict, List, Tuple

import capybara as cb
import numpy as np

from .postprocess import gen_prior_centers, get_flat_preds, get_proposals


class SCRFD:
    def __init__(
        self,
        model_path,
        batch_size: int = 1,
        score_th: float = 0.5,
        nms_th: float = 0.45,
        input_shape: tuple = None,
        **engine_kwargs,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.engine_kwargs = engine_kwargs
        self.score_th = score_th
        self.nms_th = nms_th
        self.engine = cb.ONNXEngine(
            self.model_path,
            **self.engine_kwargs,
            # session_option={'log_severity_level': 0},
            # provider_option={'enable_cuda_graph': True},
        )
        self.metadata = self.engine.metadata
        print(self.engine)

        self.inp_h, self.inp_w = self.metadata["InputSize"][2:] if input_shape is None else input_shape

        feat_sizes = [(int(self.inp_h / s), int(self.inp_w / s)) for s in self.metadata["AncStrides"]]
        self.prior_centers = gen_prior_centers(
            feat_sizes,
            self.metadata["AncStrides"],
            self.metadata["AncScales"],
        )

    def append_to_batchable(self, xs: np.ndarray):
        remaid = len(xs) % self.batch_size
        if remaid:
            xs = np.concatenate(
                [
                    xs,
                    np.zeros((self.batch_size - remaid, *xs.shape[1:]), dtype=xs.dtype),
                ],
                0,
            )
        return xs

    def detach_from_batchable(self, xs: np.ndarray, length: int):
        return xs[:length]

    def preprocess(self, imgs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        blobs, img_scales = [], []
        for img in imgs:
            img_h, img_w = img.shape[:2]
            if img_h > self.inp_h or img_w > self.inp_w:
                blob, img_scale = cb.imresize_and_pad_if_need(
                    img,
                    self.inp_h,
                    self.inp_w,
                    return_scale=True,
                    pad_value=self.metadata["Mean"],
                )
            else:
                pad_h = self.inp_h - img_h
                pad_w = self.inp_w - img_w
                blob = cb.pad(img, (0, pad_h, 0, pad_w), pad_value=self.metadata["Mean"])
                img_scale = 1
            blob = cb.imcvtcolor(blob, "BGR2RGB") if self.metadata["ColorMode"] == "rgb" else blob
            blob = (blob - self.metadata["Mean"]) / self.metadata["Std"]
            blob = blob.transpose(2, 0, 1)[None].astype("float32")
            blobs.append(blob)
            img_scales.append(img_scale)
        blobs = np.concatenate(blobs, 0)
        img_scales = np.array(img_scales)
        return blobs, img_scales

    def postprocess(self, preds: Dict[str, np.ndarray], img_scales: List[float]) -> List[dict]:
        strides = [x for x in self.metadata["AncStrides"]]
        flat_preds = get_flat_preds(preds, strides)
        proposals_list = get_proposals(
            flat_preds=flat_preds,
            prior_centers=self.prior_centers,
            img_scales=img_scales,
            score_th=self.metadata["ScoreTH"] if self.score_th is None else self.score_th,
            nms_th=self.metadata["NMSTH"] if self.nms_th is None else self.nms_th,
        )
        return proposals_list

    def __call__(self, imgs: List[np.ndarray]) -> List[dict]:
        blobs, scales = self.preprocess(imgs)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        for batch in cb.make_batch(blobs, self.batch_size):
            inputs = {name: self.append_to_batchable(batch) for name, _ in self.engine.input_infos.items()}
            tmp_preds = self.engine(**inputs)
            for k, v in tmp_preds.items():
                preds[k].append(self.detach_from_batchable(v, len(batch)))
        preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
        proposals_list = self.postprocess(preds, scales)
        return proposals_list
