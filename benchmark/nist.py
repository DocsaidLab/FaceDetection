from pprint import pprint

import capybara as cb
import pandas as pd
import torch
from fire import Fire
from torchmetrics import MeanAbsoluteError
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from src.onnx import build_onnx_pipline

DIR = cb.get_curdir(__file__)


def _info_box_to_tensor(x):
    box = x["box"]
    return torch.tensor([[box["x"], box["y"], box["w"] + box["x"], box["h"] + box["y"]]])


def _info_lmk_to_lmk5pts(x):
    lmk5pt = x["lmk5pt"]
    return torch.tensor(lmk5pt)


def _get_normalized_area(x):
    return (x["box"]["w"] * x["box"]["h"]) ** 0.5


def main(
    onnx_fpath: str,
    onnx_pipeline_name: str = "SCRFD",
    data_folder: str = "data/public/nist",
    batch_size: int = 1,
    backend: str = "cuda",
    gpu_id: int = 0,
    save_results: bool = False,
    write_to_onnx: bool = False,
):
    onnx_fpath = cb.Path(onnx_fpath)
    face_detector = build_onnx_pipline(
        onnx_pipeline_name,
        model_path=onnx_fpath,
        batch_size=batch_size,
        backend=backend,
        gpu_id=gpu_id,
    )
    results = {}
    data_folder = cb.Path(data_folder)
    sub_folders = [x for x in data_folder.glob("*") if x.is_dir()]
    for folder in sub_folders:
        jsons = cb.get_files(folder, ".json")

        mAP = MeanAveragePrecision(box_format="xyxy")
        nme = MeanAbsoluteError()

        batched_jsons = list(cb.make_batch(jsons, batch_size=batch_size))

        for batched_jsons in tqdm(batched_jsons, desc=folder.name, leave=False):
            batched_img_fpaths = [str(x).replace(".json", ".jpg") for x in batched_jsons]
            batched_imgs = [cb.imread(x) for x in batched_img_fpaths]
            proposals_list = face_detector(imgs=batched_imgs)
            for proposals, info in zip(proposals_list, batched_jsons):
                info = cb.load_json(info)
                boxes = proposals["boxes"]
                scores = proposals["scores"]
                lmk5pts = proposals["lmk5pts"]
                if len(boxes):
                    face_info = info["faces"][0]
                    gt_box = _info_box_to_tensor(face_info)
                    ious = box_iou(gt_box, torch.from_numpy(boxes)).flatten()
                    ind = torch.argmax(ious)

                    box = boxes[ind : ind + 1]
                    lmk5pt = lmk5pts[ind : ind + 1]
                    score = scores[ind]

                    pred_box = torch.from_numpy(box)
                    pred_score = torch.tensor(score)
                    label = torch.tensor([0])
                    pred_lmk5pt = torch.from_numpy(lmk5pt).flatten()
                    gt_lmk5pt = _info_lmk_to_lmk5pts(face_info).flatten()
                    normalized_area = _get_normalized_area(face_info)

                    preds = [dict(boxes=pred_box, scores=pred_score, labels=label)]
                    targets = [dict(boxes=gt_box, labels=label)]
                    mAP.update(preds, targets)
                    nme.update(pred_lmk5pt / normalized_area, gt_lmk5pt / normalized_area)
        map_results = mAP.compute()
        nme_results = nme.compute()
        tmp_results = {
            "mAP": map_results["map"].cpu().item(),
            "mAP75": map_results["map_75"].cpu().item(),
            "mAP50": map_results["map_50"].cpu().item(),
            "nme": nme_results.tolist(),
        }
        results[folder.name] = tmp_results
    results = pd.DataFrame.from_dict(results).round(4)

    out_fpath = onnx_fpath.with_suffix(".md")
    pprint(results)
    github_results = results.to_markdown()
    if save_results:
        with open(out_fpath, "w") as f:
            f.write("# Results\n\n")
            f.write(github_results)

    if write_to_onnx:
        cb.write_metadata_into_onnx(
            onnx_path=onnx_fpath,
            out_path=onnx_fpath,
            drop_old_meta=False,
            NISTEvaluation=results.to_dict(),
        )
        pprint(cb.ONNXEngine(onnx_fpath))


if __name__ == "__main__":
    Fire(main)
