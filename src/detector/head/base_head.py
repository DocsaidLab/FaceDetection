from typing import Dict, List, Tuple

import chameleon as cl
import numpy as np
import torch


class BaseHead(cl.PowerModule):
    def __init__(self, head: dict, prior: dict, loss_weight: dict, nms_cfg: dict = {}):
        super().__init__()
        self.head_cfg = head
        self.prior_cfg = prior
        self.loss_weight = loss_weight
        self.nms_cfg = nms_cfg
        self.setup_prior()
        self.setup_head()
        self.setup_loss()

    def setup_prior(self) -> None:
        raise NotImplementedError

    def setup_head(self) -> None:
        raise NotImplementedError

    def setup_loss(self) -> None:
        raise NotImplementedError

    def forward_train(
        self, feats: List[torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        feats: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @torch.inference_mode()
    def forward_to_proposals(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @torch.inference_mode()
    def forward_to_heatmaps(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @torch.inference_mode()
    def plot_assign_priors(
        self,
        img: np.ndarray,
        gts: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        raise NotImplementedError
