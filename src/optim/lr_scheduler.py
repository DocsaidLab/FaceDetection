from typing import List

from chameleon import OPTIMIZERS, MultiStepLR, WrappedLRScheduler
from torch.optim import Optimizer


@OPTIMIZERS.register_module(is_model_builder=True)
def MultiStepLRWarmUp(
    optimizer: Optimizer,
    milestones: List[int],
    warmup_milestone: int,
    gamma: float = 0.1,
    last_epoch: int = -1,
    interval='step',
    verbose: bool = False,
):
    scheduler = MultiStepLR(optimizer, milestones, gamma, last_epoch, verbose)
    return WrappedLRScheduler(
        optimizer,
        warmup_milestone,
        after_scheduler=scheduler,
        interval=interval,
    )
