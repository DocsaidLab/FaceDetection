import random
from typing import List

from chameleon import Registry
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

SAMPLERS = Registry("samplers")

SAMPLERS.register_module(module=RandomSampler)
SAMPLERS.register_module(module=SequentialSampler)


@SAMPLERS.register_module()
class MultiScaleBatchSampler(object):
    def __init__(
        self,
        dataset,
        size_candidates: List[int],
        batch_size: int,
        drop_last: bool = True,
        change_frequecy: int = 3,
        use_ddp: bool = False,
        shuffle: bool = True,
    ):
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        if change_frequecy is not None and change_frequecy < 1:
            raise ValueError(
                f"change_frequecy should be > 0, but got change_frequecy={change_frequecy}"
            )
        self.size_candidates = size_candidates
        if use_ddp:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            self.sampler = (
                RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.change_frequecy = change_frequecy

    def __iter__(self):
        size = random.choice(self.size_candidates)
        img_size = (size, size)
        batch = []
        num_batch = 0
        for idx in self.sampler:
            batch.append((idx, img_size))
            if len(batch) == self.batch_size:
                yield batch
                num_batch += 1
                batch = []
                if num_batch % self.change_frequecy == 0:
                    size = random.choice(self.size_candidates)
                    img_size = (size, size)
        if len(batch) and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
