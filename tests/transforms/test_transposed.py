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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import Transposed
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p(np.arange(5 * 4).reshape(5, 4)), [1, 0]])
    TESTS.append([p(np.arange(5 * 4).reshape(5, 4)), None])
    TESTS.append([p(np.arange(5 * 4 * 3).reshape(5, 4, 3)), [2, 0, 1]])
    TESTS.append([p(np.arange(5 * 4 * 3).reshape(5, 4, 3)), None])


class TestTranspose(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_transpose(self, im, indices):
        data = {"i": deepcopy(im), "j": deepcopy(im)}
        tr = Transposed(["i", "j"], indices)
        out_data = tr(data)
        out_im1, out_im2 = out_data["i"], out_data["j"]
        if isinstance(im, torch.Tensor):
            im = im.cpu().numpy()
        out_gt = np.transpose(im, indices)
        assert_allclose(out_im1, out_gt, type_test=False)
        assert_allclose(out_im2, out_gt, type_test=False)

        # test inverse
        fwd_inv_data = tr.inverse(out_data)
        for i, j in zip(data.values(), fwd_inv_data.values()):
            assert_allclose(i, j, type_test=False)


if __name__ == "__main__":
    unittest.main()
