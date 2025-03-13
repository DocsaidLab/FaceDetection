from functools import partial
from typing import List

import capybara as cb
from fire import Fire

from scripts.nist.make_data import make_data
from src.data.enums import FacePose
from src.data.utils import dict_to_anns

DIR = cb.get_curdir(__file__)

DataRule = {
    # VISA Images
    # ▷ The number of images is on the order of 10e5
    # ▷ The number of subjects is on the order of 10e5
    # ▷ The number of subjects with two images is on the order of 10e4
    # ▷ The images have geometry in reasonable conformance with the ISO/IEC 19794-5 Full Frontal image type. Pose is
    #   generally excellent.
    # ▷ The images are of size 252x300 pixels. The mean interocular distance(IOD) is 69 pixels.
    # ▷ The images are of subjects from greater than 100 countries, with significant imbalance due to visa issuance
    #   patterns.
    # ▷ The images are of subjects of all ages, including children, again with imbalance due to visa issuance demand.
    # ▷ Many of the images are live capture. A substantial number of the images are photographs of paper photographs.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”ISO” - see Table 4 of the
    #   FRTE API.
    # settings
    "visa": {
        "func": partial(
            make_data,
            raw_min_face_width=int(69 * 2),
            out_h=300,
            out_w=252,
            mean_face_width=int(69 * 1.7),
            std_face_width=int(69 * 1.7 * 0.2),
            max_face_width=252,
            min_face_width=69,
            pose_group=[FacePose.Frontal],
            distortion_p=0,
            blur_p=0,
        ),
        "resouce": ["data/public/celeba/origin/Frontal.txt"],
        "n_samples": 6000,
    },
    # Application Images
    # ▷ The number of images is on the order of 10e6.
    # ▷ The number of subjects is on the order of 10e6.
    # ▷ The number of subjects with two images is on the order of 10e5.
    # ▷ The images have geometry in good conformance with the ISO/IEC 19794-5 Full Frontal image.
    # ▷ The xwimages are of size 300x300 pixels. The mean interocular distance(IOD).
    # ▷ The images are of subjects from greater than 100 countries, with significant imbalance due to population and
    #   immigration patterns.
    # ▷ The images are of subjects of adults.
    # ▷ All of the images are live capture.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”WILD” - see Table 4 of the
    #   FRTE API.
    "application": {
        "func": partial(
            make_data,
            raw_min_face_width=int(61 * 2),
            out_h=300,
            out_w=300,
            mean_face_width=int(61 * 1.7),
            std_face_width=int(61 * 1.7 * 0.2),
            max_face_width=300,
            min_face_width=61,
            pose_group=[FacePose.Frontal],
            distortion_p=0,
            blur_p=0,
        ),
        "resouce": ["data/public/celeba/origin/Frontal.txt"],
        "n_samples": 6000,
    },
    # Application Images with Head Yaw
    # ▷ The number of images is on the order of 10e5.
    # ▷ The number of subjects is on the order of 10e5.
    # ▷ The number of subjects with two images is on the order of 10e5.
    # ▷ The images have geometry in good conformance with the ISO/IEC 19794-5 Full Frontal image type except the yaw
    #   angle is between 25 and 85 degrees. Our pose estimates are approximate, with an angular error that increases
    #   with yaw. The angular estimates will be improved over time.
    # ▷ The xwimages are of size 300x300 pixels. The mean interocular distance(IOD), if frontal, would be about pixels,
    #   but reduces with cosine of yaw.
    # ▷ The images are of subjects from greater than 100 countries, with significant imbalance due to population and
    #   immigration patterns.
    # ▷ The images are of subjects of adults.
    # ▷ All of the images are live capture.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”WILD” - see Table 4 of the
    #   FRTE API.
    "application_with_head_yaw": {
        "func": partial(
            make_data,
            raw_min_face_width=int(61 * 0.9 * 2),
            out_h=300,
            out_w=300,
            mean_face_width=int(61 * 0.9 * 1.7),
            std_face_width=int(61 * 0.9 * 1.7 * 0.2),
            max_face_width=300,
            min_face_width=int(61 * 0.9),
            pose_group=[
                FacePose.LeftFrontal,
                FacePose.RightFrontal,
                FacePose.LeftProfile,
                FacePose.RightProfile,
            ],
            distortion_p=0,
            blur_p=0,
        ),
        "resouce": [
            "data/public/celeba/origin/LeftFrontal.txt",
            "data/public/celeba/origin/RightFrontal.txt",
            "data/public/celeba/origin/LeftProfile.txt",
            "data/public/celeba/origin/RightProfile.txt",
        ],
        "n_samples": 6000,
    },
    # Border crossing images (過境影像)
    # ▷ The number of images is on the order of 10e6.
    # ▷ The number of subjects is on the order of 10e6.
    # ▷ The number of subjects with two images is on the order of 10e6.
    # ▷ The images are taken with at camera oriented by an attendant toward a cooperating subject. This is done under
    #   time constraints so there are role, pitch and yaw angle variations. Also background illumination is sometimes
    #   strong, so the face is under-exposed. There is some perspective distortion due to close range images. Some faces
    #   are partially cropped.
    # ▷ The images have mean IOD of 38 pixels.
    # ▷ The images are of subjects of adults and childen aged 12 or above.
    # ▷ The images are of subjects from greater than 100 countries, with significant imbalance due to population and
    #   immigration patterns.
    # ▷ The images are all live capture.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”WILD” - see Table 4 of the
    #   FRTE API.
    "border_crossing": {
        "func": partial(
            make_data,
            raw_min_face_width=int(38 * 2),
            out_h=160,
            out_w=160,
            mean_face_width=int(38 * 1.7),
            std_face_width=int(38 * 1.7 * 0.2),
            max_face_width=300,
            min_face_width=int(61 * 0.9),
            pose_group=[
                FacePose.Frontal,
                FacePose.UpFrontal,
                FacePose.DownFrontal,
                FacePose.LeftFrontal,
                FacePose.RightFrontal,
                FacePose.LeftProfile,
                FacePose.RightProfile,
            ],
            distortion_p=0.5,
            blur_p=0.5,
        ),
        "resouce": [
            "data/public/celeba/origin/Frontal.txt",
            "data/public/celeba/origin/DownFrontal.txt",
            "data/public/celeba/origin/UpFrontal.txt",
            "data/public/celeba/origin/LeftFrontal.txt",
            "data/public/celeba/origin/RightFrontal.txt",
            "data/public/celeba/origin/LeftProfile.txt",
            "data/public/celeba/origin/RightProfile.txt",
        ],
        "n_samples": 6000,
    },
    # Mugshot images (照相館照片)
    # ▷ The number of images is on the order of 10e6.
    # ▷ The number of subjects is on the order of 10e6.
    # ▷ The number of subjects with two images is on the order of 10e6.
    # ▷ The images have geometry in reasonable conformance with the ISO/IEC 19794-5 Full Frontal image type.
    # ▷ The images are of variable sizes. The median IOD is 105 pixels. The mean IOD is 113 pixels. The 1-st, 5-th, 10-th,
    #   25-th, 75-th, 90-th and 99-th percentiles are 34, 58, 70, 87, 121, 161 and 297 pixels.
    # ▷ The images are of subjects from the United States.
    # ▷ The images are of adults.
    # ▷ The images are all live capture.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”mugshot” - see Table 4 of the
    #   FRTE API.
    "mugshot": {
        "func": partial(
            make_data,
            raw_min_face_width=int(113 * 2),
            out_h=480,
            out_w=480,
            mean_face_width=int(113 * 1.7),
            std_face_width=int(113 * 1.7 * 0.5),
            max_face_width=int(297 * 1.7),
            min_face_width=int(34 * 1.7),
            pose_group=[
                FacePose.Frontal,
            ],
            distortion_p=0,
            blur_p=0,
        ),
        "resouce": [
            "data/public/celeba/origin/Frontal.txt",
        ],
        "n_samples": 6000,
    },
    # Kiosk images (自助機影像)
    # ▷ The number of images is on the order of 10e6.
    # ▷ The number of subjects is on the order of 10e5.
    # ▷ The number of subjects with multiple images is the order of 10e5.
    # ▷ The images are taken at kiosk equipped with a camera intended to capture a centered face. However the images
    #   have specific quality defects arising from the camera triggering before the subject looks at it. These are
    #   downward pitch of the face relative to the optical axis; cropping of the forehead; and cropping of left or right
    #   part of the face. Partial cropping affects perhaps 10% of the images. Resolution does not vary widely.
    # ▷ The images are of adults.
    # ▷ The images have mean IOD of 44 pixels, with maximum below 75, and minimum when both eyes are present
    #   above 25 pixels.
    # ▷ All of the images are live capture, none are scanned.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”WILD” - see Table 4 of the FRTE
    #   API.
    "kiosk": {
        "func": partial(
            make_data,
            raw_min_face_width=int(44 * 2),
            out_h=160,
            out_w=160,
            mean_face_width=int(44 * 1.7),
            std_face_width=int(44),
            max_face_width=int(75 * 1.7),
            min_face_width=int(25 * 1.7),
            pose_group=[
                FacePose.Frontal,
                FacePose.UpFrontal,
                FacePose.DownFrontal,
                FacePose.LeftFrontal,
                FacePose.RightFrontal,
            ],
            distortion_p=1,
            blur_p=1,
        ),
        "resouce": [
            "data/public/celeba/origin/Frontal.txt",
            "data/public/celeba/origin/DownFrontal.txt",
            "data/public/celeba/origin/UpFrontal.txt",
            "data/public/celeba/origin/LeftFrontal.txt",
            "data/public/celeba/origin/RightFrontal.txt",
        ],
        "n_samples": 6000,
    },
    # Wild images (野外影像)
    # ▷ The number of images is on the order of 10e5.
    # ▷ The number of subjects is on the order of 10e4.
    # ▷ The number of subjects with two images on the order of 10e4.
    # ▷ The images include many photojournalism-style images. Images are given to the algorithm using a variable but
    #   generally tight crop of the head. Resolution varies very widely. The images are very unconstrained, with wide yaw
    #   and pitch pose variation. Faces can be occluded, including hair and hands.
    # ▷ The images are of adults.
    # ▷ All of the images are live capture, none are scanned.
    # ▷ When these images are input to the algorithm, they are labelled as being of type ”WILD” - see Table 4 of the FRTE
    #   API.
    "wild": {
        "func": partial(
            make_data,
            raw_min_face_width=int(24 * 2),
            out_h=160,
            out_w=160,
            mean_face_width=int(44 * 1.7),
            std_face_width=int(44 * 1.2),
            max_face_width=int(75 * 1.7),
            min_face_width=int(25 * 1.7),
            pose_group=[
                FacePose.Frontal,
                FacePose.UpFrontal,
                FacePose.DownFrontal,
                FacePose.LeftFrontal,
                FacePose.RightFrontal,
                FacePose.LeftProfile,
                FacePose.RightProfile,
            ],
            distortion_p=1,
            blur_p=1,
        ),
        "resouce": ["data/public/widerface_train/val.txt"],
        "n_samples": 6000,
    },
}


# def _get_norm_size(mean=69, std=69 * 0.15, max=np.inf, min=20):
#     return np.clip(np.random.normal(loc=mean, scale=std), a_max=max, a_min=min)


# def _get_is_valid_mask_and_scales(faces, datarule):
#     target_pose = datarule['pose']
#     boxes = np.stack([x.numpy() for x in faces.box])
#     lmk5pts = np.stack([x.numpy() if x is not None else np.full((5, 2), -1) for x in faces.lmk5pt])
#     poses = np.array([x.value if x is not None else -1 for x in faces.pose])

#     if datarule['min_iod'] is None:
#         face_ws = boxes[:, 2]
#         flag1 = (face_ws >= datarule['min_box_width']).flatten()
#         flag2 = (poses == target_pose).flatten() if target_pose is not None else flag1
#         dst_scales = np.array([_get_norm_size(**datarule['scale']) / (x + 1e-8) for x in face_ws])
#     else:
#         face_iods = ((lmk5pts[:, 0] - lmk5pts[:, 1]) ** 2).sum(-1) ** 0.5
#         flag1 = (face_iods >= datarule['min_iod']).flatten()
#         flag2 = (poses == target_pose).flatten() if target_pose is not None else flag1
#         dst_scales = np.array([_get_norm_size(**datarule['scale']) / (x + 1e-8) for x in face_iods])
#     mask = np.stack((flag1, flag2), -1).all(-1)
#     inds = np.argwhere(mask).flatten().tolist()

#     return dst_scales[inds], inds


# def get_cropped_faces(faces: aipkg.Faces, datarule: dict) -> List[aipkg.Faces]:
#     # get valid iod and inds from unzip_anns
#     dst_scales, inds = _get_is_valid_mask_and_scales(faces, datarule)
#     face_list = [faces[i] for i in inds]
#     new_faces_list = []
#     img = faces.raw_image
#     for face, dst_scale in zip(face_list, dst_scales):
#         box, lmk5pt, pose = face.box.convert('XYXY').numpy(), face.lmk5pt.numpy(), face.pose
#         dst_img = cv2.resize(img.copy(), None, fx=dst_scale, fy=dst_scale, interpolation=cv2.INTER_AREA)
#         box = box * dst_scale
#         lmk5pt = lmk5pt * dst_scale if lmk5pt[0, 0] != -1 else lmk5pt
#         box_center = _get_box_center(box)
#         dst_h, dst_w = datarule['image_size']
#         cropped_box = acv.Box(np.array((*box_center, dst_w, dst_h)), "CXCYWH")
#         delta_xy = cropped_box.left_top.tolist()
#         dst_img = acv.imcropbox(dst_img, cropped_box, with_padding=True,)
#         box -= delta_xy * 2
#         lmk5pt = lmk5pt - delta_xy if lmk5pt[0, 0] != -1 else lmk5pt

#         box = acv.Box(box)
#         lmk5pt = acv.Keypoints(lmk5pt)

#         new_faces_list.append(
#             aipkg.Faces(
#                 raw_image=dst_img,
#                 faces=[
#                     aipkg.Face(
#                         box=box,
#                         lmk5pt=lmk5pt,
#                         pose=pose,
#                     )
#                 ]
#             )
#         )

#     return new_faces_list


def load_data(resouce: List[str]):
    data = []
    for res in resouce:
        folder = cb.Path(res).parent
        with open(res, "r") as f:
            jsons = [folder / x.strip() for x in f.readlines()]
            tmp_data = []
            for j in jsons:
                tmp = cb.load_json(j)
                tmp["img_fpath"] = j.parent / tmp["img_fname"]
                tmp_data.append(tmp)
            data.extend(tmp_data)
    return data


def main(out_folder: str = "data/nist2", preview: bool = False):
    out_folder = cb.Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    for k, v in DataRule.items():
        sub_folder = out_folder / k / "data"
        sub_folder.mkdir(exist_ok=True, parents=True)

        if preview:
            preview_folder = out_folder / k / "preview"
            preview_folder.mkdir(exist_ok=True, parents=True)

        data = load_data(v["resouce"])
        samples = v["func"](data, v["n_samples"])
        txt_fpath = out_folder / k / "all.txt"
        anns_fpaths = []
        for i, sample in enumerate(samples):
            save_folder = sub_folder / f"{i // 1000 * 1000:06d}"
            save_folder.mkdir(exist_ok=True, parents=True)

            img_fpath = save_folder / f"{i:06d}.jpg"
            anns_fpath = save_folder / f"{i:06d}.json"
            cb.imwrite(sample["img"], img_fpath)
            cb.dump_json(sample["anns"], anns_fpath)
            anns_fpath = anns_fpath.relative_to(sub_folder.parent)
            anns_fpaths.append(anns_fpath)

            if preview:
                preview_save_folder = preview_folder / f"{i // 1000 * 1000:06d}"
                preview_save_folder.mkdir(exist_ok=True, parents=True)
                anns = dict_to_anns(sample["anns"])
                plotted = cb.draw_keypoints_list(
                    cb.draw_boxes(sample["img"], anns["boxes"], thicknesses=1),
                    anns["lmk5pts"],
                    scales=0.2,
                )
                cb.imwrite(plotted, preview_save_folder / f"{i:06d}.jpg")
        with open(txt_fpath, "w") as f:
            f.write("\n".join([str(x) for x in anns_fpaths]))


if __name__ == "__main__":
    Fire(main)
