# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence

import torch
from torch.utils.data import DistributedSampler as _TorchDistributedSampler

__all__ = ["DistributedSampler", "DistributedWeightedRandomSampler"]


class DistributedSampler(_TorchDistributedSampler):
    """
    Enhance PyTorch DistributedSampler to support non-evenly divisible sampling.

    Args:
        even_divisible: if False, different ranks can have different data length.
        for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].
        args: additional arguments for `DistributedSampler` super class.
        kwargs: additional arguments for `DistributedSampler` super class.

    More information about DistributedSampler, please check:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py

    """

    def __init__(self, even_divisible: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not even_divisible:
            data_len = len(kwargs["dataset"])
            extra_size = self.total_size - data_len
            if self.rank + extra_size >= self.num_replicas:
                self.num_samples -= 1
            self.total_size = data_len


class DistributedWeightedRandomSampler(DistributedSampler):
    """
    Extend the `DistributedSampler` to support weighted sampling.
    Refer to `torch.utils.data.WeightedRandomSampler`, for more details please check:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py#L150

    Args:
        weights: a sequence of weights, not necessary summing up to one, length should exactly
            match the full dataset.
        num_samples_per_rank: number of samples to draw for every rank, sample from
            the distributed subset of dataset.
            if None, default to the length of dataset split by DistributedSampler.
        replacement: if ``True``, samples are drawn with replacement, otherwise, they are
            drawn without replacement, which means that when a sample index is drawn for a row,
            it cannot be drawn again for that row, default to True.
        generator: PyTorch Generator used in sampling.
        even_divisible: if False, different ranks can have different data length.
            for example, input data: [1, 2, 3, 4, 5], rank 0: [1, 3, 5], rank 1: [2, 4].'
        args: additional arguments for `DistributedSampler` super class.
        kwargs: additional arguments for `DistributedSampler` super class.

    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples_per_rank: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
        even_divisible: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(even_divisible, *args, **kwargs)
        self.weights = weights
        self.num_samples_per_rank = num_samples_per_rank
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        indices = list(super().__iter__())
        num_samples = self.num_samples_per_rank if self.num_samples_per_rank is not None else self.num_samples
        weights = torch.as_tensor([self.weights[i] for i in indices], dtype=torch.double)
        # sample based on the provided weights
        rand_tensor = torch.multinomial(weights, num_samples, self.replacement, generator=self.generator)

        for i in rand_tensor:
            yield indices[i]
