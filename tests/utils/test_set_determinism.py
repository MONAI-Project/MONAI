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

from monai.utils import get_seed, set_determinism
from tests.util import SkipIfBeforePyTorchVersion, skip_if_no_cuda


class TestSetDeterminism(unittest.TestCase):
    def test_values(self):
        # check system default flags
        set_determinism(None)
        self.assertTrue(not torch.backends.cudnn.deterministic)
        self.assertTrue(get_seed() is None)
        # set default seed
        set_determinism()
        self.assertTrue(get_seed() is not None)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertTrue(not torch.backends.cudnn.benchmark)
        # resume default
        set_determinism(None)
        self.assertTrue(not torch.backends.cudnn.deterministic)
        self.assertTrue(not torch.backends.cudnn.benchmark)
        self.assertTrue(get_seed() is None)
        # test seeds
        seed = 255
        set_determinism(seed=seed)
        self.assertEqual(seed, get_seed())
        a = np.random.randint(seed)
        b = torch.randint(seed, (1,))
        # tset when global flag support is disabled
        torch.backends.disable_global_flags()
        set_determinism(seed=seed)
        c = np.random.randint(seed)
        d = torch.randint(seed, (1,))
        self.assertEqual(a, c)
        self.assertEqual(b, d)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertTrue(not torch.backends.cudnn.benchmark)
        set_determinism(seed=None)


class TestSetFlag(unittest.TestCase):
    def setUp(self):
        set_determinism(1, use_deterministic_algorithms=True)

    @SkipIfBeforePyTorchVersion((1, 8))  # beta feature
    @skip_if_no_cuda
    def test_algo(self):
        with self.assertRaises(RuntimeError):
            x = torch.randn(20, 16, 50, 44, 31, requires_grad=True, device="cuda:0")
            y = torch.nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))(x)
            y.sum().backward()

    def tearDown(self):
        set_determinism(None, use_deterministic_algorithms=False)


if __name__ == "__main__":
    unittest.main()
