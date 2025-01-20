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

import torch

from monai.data import CacheDataset, ThreadDataLoader
from monai.transforms import ToDeviced
from tests.util import assert_allclose, skip_if_no_cuda


@skip_if_no_cuda
class TestToDeviced(unittest.TestCase):
    def test_value(self):
        device = "cuda:0"
        data = [{"img": torch.tensor(i)} for i in range(4)]
        dataset = CacheDataset(
            data=data, transform=ToDeviced(keys="img", device=device, non_blocking=True), cache_rate=1.0
        )
        dataloader = ThreadDataLoader(dataset=dataset, num_workers=0, batch_size=1)
        for i, d in enumerate(dataloader):
            assert_allclose(d["img"], torch.tensor([i], device=device))


if __name__ == "__main__":
    unittest.main()
