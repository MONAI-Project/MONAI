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
from parameterized import parameterized

from monai.transforms import Transpose
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p(np.arange(5 * 4).reshape(5, 4)), None])
    TESTS.append([p(np.arange(5 * 4 * 3).reshape(5, 4, 3)), [2, 0, 1]])


class TestTranspose(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_transpose(self, im, indices):
        tr = Transpose(indices)
        out1 = tr(im)
        if isinstance(im, torch.Tensor):
            im = im.cpu().numpy()
        out2 = np.transpose(im, indices)
        assert_allclose(out1, out2, type_test=False)


if __name__ == "__main__":
    unittest.main()
