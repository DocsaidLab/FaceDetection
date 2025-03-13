from copy import copy
from typing import Dict, List, Tuple

import capybara as cb
import chameleon as cl
import torch
import torch.nn as nn
from onnxsim import model_info, simplify

import onnx

from ..utils.draw import draw_results
from .backbone import BACKBONES
from .head import HEADS
from .neck import NECKS


class Detector(nn.Module):
    def __init__(
        self,
        backbone_dict: dict,
        neck_dict: dict,
        head_dict: dict,
        onnx_dict: dict = {},
    ):
        super().__init__()
        self.backbone = BACKBONES.build(backbone_dict)
        self.neck = NECKS.build(neck_dict)
        self.head = HEADS.build(head_dict)
        self.onnx_dict = onnx_dict

    def forward_feats(self, xs) -> List[torch.Tensor]:
        dtype = xs.dtype
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not torch.onnx.is_in_onnx_export()):
            feats = self.backbone(xs)
            feats = self.neck(feats)
        feats = [x.to(dtype) for x in feats]
        return feats

    def forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self.forward_feats(batch["image"])
        losses = self.head.forward_train(feats, batch)
        return losses

    def forward(
        self,
        xs: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        feats = self.forward_feats(xs)
        level_results = self.head(feats)
        return level_results

    @torch.inference_mode()
    def forward_to_proposals(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        feats = self.forward_feats(xs)
        return self.head.forward_to_proposals(feats)

    @torch.inference_mode()
    def forward_to_heatmaps(
        self,
        xs: torch.Tensor,
    ) -> List[torch.Tensor]:
        feats = self.forward_feats(xs)
        score_maps = self.head.forward_to_heatmaps(feats)
        return score_maps

    @torch.inference_mode()
    def demo_one_img(self, img):
        img = cb.imcvtcolor(img, "BGR2RGB")
        xs = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        boxes, scores, lmk5pts = self.forward_to_proposals(xs=xs)
        boxes = boxes[0].numpy()
        scores = scores[0].numpy().flatten()
        kpts_list = lmk5pts[0].numpy().reshape(-1, 5, 2)
        img = draw_results(img, boxes, scores, kpts_list, draw_scale=2)
        return img

    @torch.inference_mode()
    def to_onnx(
        self,
        onnx_path: str,
        verbose: bool = False,
        overwrite_export_kwargs: dict = {},
        overwrite_metadata_kwargs: dict = {},
    ):
        self.eval()
        input_size = tuple(self.onnx_dict["metadata"]["InputSize"])
        _, macs, params = cl.calculate_flops(self, input_size, print_detailed=False)

        inps = torch.rand(input_size).type_as(next(self.parameters()))
        export_kwargs = copy(self.onnx_dict.get("export", {}))
        export_kwargs.update(overwrite_export_kwargs)
        torch.onnx.export(
            self,
            inps,
            onnx_path,
            **export_kwargs,
        )
        onnx_model = onnx.load(onnx_path)
        sim_model, _ = simplify(onnx_model)
        onnx.save(sim_model, onnx_path)
        model_info.print_simplifying_info(onnx_model, sim_model)

        metadata_kwargs = copy(self.onnx_dict.get("metadata", {}))
        metadata_kwargs.update(overwrite_metadata_kwargs)
        metadata_kwargs["Macs"] = macs
        metadata_kwargs["Params"] = params
        cb.write_metadata_into_onnx(
            onnx_path,
            onnx_path,
            **metadata_kwargs,
        )
        if verbose:
            print(cb.ONNXEngine(onnx_path, backend="cpu"))

        return onnx_path
