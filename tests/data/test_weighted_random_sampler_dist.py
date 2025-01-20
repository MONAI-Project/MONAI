# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.distributed as dist

from monai.data import DistributedWeightedRandomSampler
from tests.util import DistCall, DistTestCase, skip_if_darwin, skip_if_windows


@skip_if_windows
@skip_if_darwin
class DistributedWeightedRandomSamplerTest(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_sampling(self):
        data = [1, 2, 3, 4, 5]
        weights = [1, 2, 3, 4, 5]
        sampler = DistributedWeightedRandomSampler(
            weights=weights, dataset=data, shuffle=False, generator=torch.Generator().manual_seed(0)
        )
        samples = np.array([data[i] for i in list(sampler)])

        if dist.get_rank() == 0:
            np.testing.assert_allclose(samples, np.array([5, 5, 5]))

        if dist.get_rank() == 1:
            np.testing.assert_allclose(samples, np.array([1, 4, 4]))

    @DistCall(nnodes=1, nproc_per_node=2)
    def test_num_samples(self):
        data = [1, 2, 3, 4, 5]
        weights = [1, 2, 3, 4, 5]
        sampler = DistributedWeightedRandomSampler(
            weights=weights,
            num_samples_per_rank=5,
            dataset=data,
            shuffle=False,
            generator=torch.Generator().manual_seed(123),
        )
        samples = np.array([data[i] for i in list(sampler)])

        if dist.get_rank() == 0:
            np.testing.assert_allclose(samples, np.array([3, 1, 5, 1, 5]))

        if dist.get_rank() == 1:
            np.testing.assert_allclose(samples, np.array([4, 2, 4, 2, 4]))


if __name__ == "__main__":
    unittest.main()
