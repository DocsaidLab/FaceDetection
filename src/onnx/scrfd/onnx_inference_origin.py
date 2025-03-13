import json
from typing import Dict, List, Tuple

import capybara as cb
import numpy as np

from .postprocess import gen_prior_centers, get_proposals


class SCRFD:
    def __init__(
        self,
        model_path,
        batch_size: int = 1,
        score_th: float = 0.5,
        nms_th: float = 0.45,
        input_shape: tuple = None,
        mean_value: float = 127.5,
        std_value: float = 128,
        **engine_kwargs,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.engine_kwargs = engine_kwargs
        self.score_th = score_th
        self.nms_th = nms_th
        self.engine = cb.ONNXEngine(self.model_path, **self.engine_kwargs)
        default_metadata = {
            "ColorMode": "rgb",
            "InputSize": [1, 3, 640, 640],
            "AncStrides": [
                [8, 8],
                [16, 16],
                [32, 32],
            ],
            "AncScales": [1, 2],
            "NMSTH": 0.45,
            "ScoreTH": 0.02,
        }
        metadata = self.engine.metadata.get("metadata", None)
        self.metadata = (
            json.loads(metadata) if metadata is not None else default_metadata
        )
        print(self.engine)

        self.mean_value = mean_value
        self.std_value = std_value

        self.inp_h, self.inp_w = (
            self.metadata["InputSize"][2:] if input_shape is None else input_shape
        )

        feat_sizes = [
            (int(self.inp_h / s[0]), int(self.inp_w / s[1]))
            for s in self.metadata["AncStrides"]
        ]
        self.prior_centers = gen_prior_centers(
            feat_sizes,
            self.metadata["AncStrides"],
            self.metadata["AncScales"],
        )

        # input_name = {
        #     np.zeros((self.batch_size, 3, self.inp_h, self.inp_w), dtype='float32')
        #     for name, info in self.engine.input_infos.items()
        # }
        # self.engine(**{input_name: mock_inputs})

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
            blob, img_scale = cb.imresize_and_pad_if_need(
                img, self.inp_h, self.inp_w, return_scale=True
            )
            blob = (
                cb.imcvtcolor(blob, "BGR2RGB")
                if self.metadata["ColorMode"] == "rgb"
                else blob
            )
            blob = (blob - self.mean_value) / self.std_value
            blob = blob.transpose(2, 0, 1)[None].astype("float32")
            blobs.append(blob)
            img_scales.append(img_scale)
        blobs = np.concatenate(blobs, 0)
        img_scales = np.array(img_scales)
        return blobs, img_scales

    def postprocess(
        self, preds: Dict[str, np.ndarray], img_scales: List[float]
    ) -> List[dict]:
        obj_preds = np.concatenate(
            [preds[k] for k in preds.keys() if "score" in k], axis=1
        )
        loc_preds = np.concatenate(
            [preds[k] for k in preds.keys() if "box" in k], axis=1
        )
        lmk_preds = np.concatenate(
            [preds[k] for k in preds.keys() if "kps" in k], axis=1
        )
        flat_preds = (loc_preds, obj_preds, lmk_preds)
        proposals_list = get_proposals(
            flat_preds=flat_preds,
            prior_centers=self.prior_centers,
            img_scales=img_scales,
            score_th=self.metadata["ScoreTH"]
            if self.score_th is None
            else self.score_th,
            nms_th=self.metadata["NMSTH"] if self.nms_th is None else self.nms_th,
        )
        return proposals_list

    def __call__(self, imgs: List[np.ndarray]) -> List[dict]:
        blobs, scales = self.preprocess(imgs)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        for batch in cb.make_batch(blobs, self.batch_size):
            inputs = {
                name: self.append_to_batchable(batch)
                for name, _ in self.engine.input_infos.items()
            }
            tmp_preds = self.engine(**inputs)
            for k, v in tmp_preds.items():
                preds[k].append(self.detach_from_batchable(v, len(batch)))
        preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
        proposals_list = self.postprocess(preds, scales)
        return proposals_list
