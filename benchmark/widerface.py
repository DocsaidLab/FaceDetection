import os
from itertools import chain
from pathlib import Path
from pprint import pprint

import capybara as cb
import numpy as np
from fire import Fire
from mmdet.evaluation import bbox_overlaps
from scipy.io import loadmat
from tabulate import tabulate
from tqdm import tqdm

from src.onnx import build_onnx_pipline


def get_gt_boxes(gt_dir) -> dict:
    gt_mat = loadmat(os.path.join(gt_dir, "wider_face_val.mat"))
    hard_mat = loadmat(os.path.join(gt_dir, "wider_hard_val.mat"))
    medium_mat = loadmat(os.path.join(gt_dir, "wider_medium_val.mat"))
    easy_mat = loadmat(os.path.join(gt_dir, "wider_easy_val.mat"))

    boxes_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    files_list = gt_mat["file_list"]

    hard_gt_list = hard_mat["gt_list"]
    medium_gt_list = medium_mat["gt_list"]
    easy_gt_list = easy_mat["gt_list"]

    gts_dict = {}
    zipped = zip(boxes_list, event_list, files_list, hard_gt_list, medium_gt_list, easy_gt_list)
    for boxes, event, files, hards, mediums, easys in zipped:
        boxes = [x[0] for x in boxes[0]]
        files = list(chain(*[[x[0][0].tolist() for x in xs] for xs in files]))
        event = event[0][0]
        hards = [x[0].flatten() for x in hards[0]]
        mediums = [x[0].flatten() for x in mediums[0]]
        easys = [x[0].flatten() for x in easys[0]]
        gts_dict.update(**{
            f"{event}/{file}": {"boxes": b, "hard": h, "medium": m, "easy": e}
            for file, b, h, m, e in zip(files, boxes, hards, mediums, easys)
        })

    return gts_dict


def image_eval(pred, gt, ignore, iou_thresh):
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(len(_pred)):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype(float)
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / (pr_curve[i, 0] + 1e-9)
        _pr_curve[i, 1] = pr_curve[i, 1] / (count_face + 1e-9)
    return _pr_curve


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def norm_score(preds_dict):
    """norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """
    max_score = 0
    min_score = 1

    for _, value in preds_dict.items():
        if len(value) == 0:
            continue
        _min = np.min(value[:, -1])
        _max = np.max(value[:, -1])
        max_score = max(_max, max_score)
        min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, value in preds_dict.items():
        if len(value) == 0:
            continue
        value[:, -1] = (value[:, -1] - min_score) / (diff + 1e-9)


def run_with_onnx(onnx_service, data_folder) -> dict:
    testset_folder = f"{data_folder}"
    preds_dict = {}
    files = cb.get_files(testset_folder, suffix=[".jpg", ".jpeg", ".png"])
    for image_fpath in tqdm(
        files,
        "Widerface Evaluation",
        ncols=80,
        total=len(files),
        leave=False,
    ):
        image_fpath = Path(image_fpath)
        img = cb.imread(image_fpath)
        fpath = "/".join(Path(image_fpath).parts[-2:])
        fpath = fpath.split(".")[0]
        if img is None:
            print(f"ignore : {image_fpath}")
            continue
        proposals = onnx_service(imgs=[img])[0]
        if len(proposals):
            boxes = cb.Boxes(proposals["boxes"]).convert("XYWH").numpy()
            scores = proposals["scores"]
            pred = np.concatenate((boxes, scores), -1, dtype="float")
        else:
            pred = []
        preds_dict[fpath] = pred
    return preds_dict


def gen_aps(
    preds_dict,
    gt_folder="data/public/widerface_evaluation/gt",
):
    thresh_num = 1000
    gts_dict = get_gt_boxes(gt_folder)
    norm_score(preds_dict)
    aps = {}
    for setting in ["easy", "medium", "hard"]:
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype(float)
        for k, gt_dict in tqdm(
            gts_dict.items(),
            setting,
            ncols=80,
            total=len(gts_dict),
        ):
            preds = preds_dict[k]
            gt_inds = gt_dict[setting]
            gt_boxes = gt_dict["boxes"]
            count_face += len(gt_inds)
            if len(gt_boxes) == 0 or len(preds) == 0:
                continue
            ignore = np.zeros(gt_boxes.shape[0])
            if len(gt_inds) != 0:
                ignore[gt_inds - 1] = 1
            pred_recall, proposal_list = image_eval(preds, gt_boxes, ignore, 0.5)
            pr_curve += img_pr_info(thresh_num, preds, proposal_list, pred_recall)
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps[setting] = [round(ap, 3).item()]

    tabulate_aps = tabulate(aps, headers="keys", tablefmt="github")
    aps_out = {k: v[0] for k, v in aps.items()}
    return tabulate_aps, aps_out


def main(
    onnx_fpath: str,
    onnx_pipeline_name: str = "SCRFD",
    data_folder: str = "data/public/widerface_evaluation",
    save_results: bool = False,
    write_to_onnx: bool = False,
):
    onnx_fpath = Path(onnx_fpath)
    onnx_pipline = build_onnx_pipline(
        name=onnx_pipeline_name,
        model_path=onnx_fpath,
        score_th=0.02,
        nms_th=0.45,
        backend="cuda",
        gpu_id=0,
    )
    data_folder = Path(data_folder)
    preds_dict = run_with_onnx(onnx_pipline, data_folder / "images")
    tabulate_aps, aps_out = gen_aps(preds_dict, data_folder / "gt")
    print(tabulate_aps)
    if save_results is not None:
        cb.dump_json(aps_out, Path(onnx_fpath.parent, f"{onnx_fpath.stem}_widerface.json"))

    if write_to_onnx:
        cb.write_metadata_into_onnx(
            onnx_path=onnx_fpath,
            out_path=onnx_fpath,
            drop_old_meta=False,
            WiderfaceEvaluation=aps_out,
        )
        pprint(cb.ONNXEngine(onnx_fpath))
    return tabulate_aps, aps_out


if __name__ == "__main__":
    Fire(main)
