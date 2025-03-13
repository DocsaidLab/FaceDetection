import random
from typing import List

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint


def restore_from_ckpt(model: nn.Module, model_ckpt: dict):
    ori_state_dict = model.state_dict()
    unexpected = []

    for k, v in model_ckpt.items():
        target = ori_state_dict.get(k, None)
        if target is not None and target.shape == v.shape:
            ori_state_dict[k] = v
        else:
            if target is not None:
                unexpected.append(f"ckpt's {k} = {v.shape}, but model's {k} = {target.shape}")
            else:
                unexpected.append(f"target of {k} is None")

    cprint(text=f'Unexpected key and value: \n{unexpected}', color='red')

    sucessful = model.load_state_dict(ori_state_dict, strict=True)

    if not sucessful:
        raise KeyError(f'load_state_dict has an error.')

    cprint('Load sucessfully...', 'yellow')


def get_model_params(model: nn.Module, excludes: List[str] = []):
    model_params = {
        name: params for name, params in model.named_parameters()
    }
    for name, _ in model.named_parameters():
        for exclude in excludes:
            if exclude in name:
                model_params.pop(name)
    return iter(list(model_params.values()))


def setup_opencv():
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(0)


def setup_torch():
    import os
    os.environ['OMP_NUM_THREADS'] = "1"
    torch.set_num_threads(8)
    torch.set_printoptions(4, sci_mode=False)
    torch.set_float32_matmul_precision('high')


def setup_seed(seed):
    L.seed_everything(seed, verbose=False)
