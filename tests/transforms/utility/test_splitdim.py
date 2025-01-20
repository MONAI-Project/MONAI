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
from parameterized import parameterized

from monai.transforms.utility.array import SplitDim
from tests.util import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    for keepdim in (True, False):
        TESTS.append(((2, 10, 8, 7), keepdim, p))


class TestSplitDim(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_correct_shape(self, shape, keepdim, im_type):
        arr = im_type(np.random.rand(*shape))
        for dim in range(arr.ndim):
            out = SplitDim(dim, keepdim)(arr)
            self.assertIsInstance(out, (list, tuple))
            self.assertEqual(type(out[0]), type(arr))
            self.assertEqual(len(out), arr.shape[dim])
            expected_ndim = arr.ndim if keepdim else arr.ndim - 1
            self.assertEqual(out[0].ndim, expected_ndim)
            # assert is a shallow copy
            arr[0, 0, 0, 0] *= 2
            self.assertEqual(arr.flatten()[0], out[0].flatten()[0])

    def test_singleton(self):
        shape = (2, 1, 8, 7)
        for p in TEST_NDARRAYS:
            arr = p(np.random.rand(*shape))
            out = SplitDim(dim=1)(arr)
            self.assertEqual(out[0].shape, shape)


if __name__ == "__main__":
    unittest.main()
