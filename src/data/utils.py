from typing import Any, Dict, Union

import capybara as cb
import numpy as np

from .enums import FacePose


def dict_to_anns(d: Dict[str, list]) -> Dict[str, Any]:
    faces = d["faces"]
    n_faces = len(faces)
    boxes = np.full((n_faces, 4), -1, dtype="float32")
    lmk5pts = np.full((n_faces, 5, 2), -1, dtype="float32")
    # lmk106pts = np.full((n_faces, 106, 2), -1, dtype='float32')
    has_lmk5pts = np.zeros((n_faces, 1), dtype="float32")
    poses = np.full((n_faces, 1), -1, dtype="float32")

    for i, face in enumerate(faces):
        box = face["box"]
        boxes[i] = [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]]
        lmk5pts[i] = face["lmk5pt"] if "lmk5pt" in face else -1
        # lmk106pts[i] = face['lmk106pt'] if 'lmk106pt' in face else -1
        poses[i] = getattr(FacePose, face["pose"]).value if "pose" in face else -1
        has_lmk5pts[i] = (lmk5pts[i][0, 0] != -1) * 1

    return {
        "boxes": boxes,
        "lmk5pts": lmk5pts,
        # 'lmk106pts': lmk106pts,
        "has_lmk5pts": has_lmk5pts,
        "poses": poses,
        "box_format": "XYXY",
    }


def anns_to_dict(anns: Dict[str, np.ndarray]) -> Dict[str, Any]:
    boxes = anns["boxes"]
    lmk5pts = anns["lmk5pts"]
    # lmk106pts = anns['lmk106pts']
    has_lmk5pts = anns["has_lmk5pts"]
    poses = anns["poses"]
    box_format = anns["box_format"]

    faces = []
    for i in range(boxes.shape[0]):
        face = {
            "box": {
                "x": boxes[i][0],
                "y": boxes[i][1],
                "w": boxes[i][2] - boxes[i][0],
                "h": boxes[i][3] - boxes[i][1],
            },
            "pose": FacePose(poses[i]).name if poses[i] != -1 else None,
        }
        if has_lmk5pts[i] == 1:
            face["lmk5pt"] = lmk5pts[i]
        faces.append(face)

    return {"faces": faces, "box_format": box_format}


def oneface_to_dict(
    box: Union[np.ndarray, cb.Box],
    lmk5pt: Union[np.ndarray, cb.Box],
    pose: Union[int, str, FacePose],
) -> Dict[str, Any]:
    box = box if isinstance(box, np.ndarray) else box.numpy()
    box = box.tolist()
    lmk5pt = lmk5pt if isinstance(lmk5pt, np.ndarray) else lmk5pt.numpy()
    lmk5pt = lmk5pt.tolist()
    pose = FacePose.obj_to_enum(pose)
    return {
        "faces": [
            {
                "box": {
                    "x": box[0],
                    "y": box[1],
                    "w": box[2] - box[0],
                    "h": box[3] - box[1],
                },
                "lmk5pt": lmk5pt,
                "pose": pose.name,
            }
        ],
        "box_format": "XYXY",
    }
