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

from monai.data import DataLoader
from monai.utils import set_determinism
from tests.util import DistTestCase, SkipIfBeforePyTorchVersion, TimedCall, skip_if_no_cuda, skip_if_quick


def run_loading_test(num_workers=50, device=None, pw=False):
    """multi workers stress tests"""
    set_determinism(seed=0)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_ds = list(range(10000))
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=num_workers, persistent_workers=pw)
    answer = []
    for _ in range(2):
        np.testing.assert_equal(torch.cuda.memory_allocated(), 0)
        for batch_data in train_loader:
            x = batch_data.to(device)
            mem = torch.cuda.memory_allocated()
            np.testing.assert_equal(mem > 0 and mem < 5000, True)
        answer.append(x[-1].item())
        del x
    return answer


@skip_if_quick
@skip_if_no_cuda
@SkipIfBeforePyTorchVersion((1, 9))
class IntegrationLoading(DistTestCase):
    def tearDown(self):
        set_determinism(seed=None)

    @TimedCall(seconds=5000, skip_timing=not torch.cuda.is_available(), daemon=False)
    def test_timing(self):
        expected = None
        for pw in (False, True):
            result = run_loading_test(pw=pw)
            if expected is None:
                expected = result[0]
        np.testing.assert_allclose(result[0], expected)  # test for deterministic first epoch in two settings


if __name__ == "__main__":
    unittest.main()
