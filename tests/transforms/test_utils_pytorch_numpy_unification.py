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

from monai.transforms.utils_pytorch_numpy_unification import max, min, mode, percentile
from monai.utils import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose, skip_if_quick

TEST_MODE = []
for p in TEST_NDARRAYS:
    TEST_MODE.append([p(np.array([1, 2, 3, 4, 4, 5])), p(4), False])
    TEST_MODE.append([p(np.array([3.1, 4.1, 4.1, 5.1])), p(4.1), False])
    TEST_MODE.append([p(np.array([3.1, 4.1, 4.1, 5.1])), p(4), True])

TEST_MIN_MAX = []
for p in TEST_NDARRAYS:
    TEST_MIN_MAX.append([p(np.array([1, 2, 3, 4, 4, 5])), {}, min, p(1)])
    TEST_MIN_MAX.append([p(np.array([[3.1, 4.1, 4.1, 5.1], [3, 5, 4.1, 5]])), {"dim": 1}, min, p([3.1, 3])])
    TEST_MIN_MAX.append([p(np.array([1, 2, 3, 4, 4, 5])), {}, max, p(5)])
    TEST_MIN_MAX.append([p(np.array([[3.1, 4.1, 4.1, 5.1], [3, 5, 4.1, 5]])), {"dim": 1}, max, p([5.1, 5])])


class TestPytorchNumpyUnification(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(0)

    def test_percentile(self):
        for size in (1, 100):
            q = np.random.randint(0, 100, size=size)
            results = []
            for idx, p in enumerate(TEST_NDARRAYS):
                dtype = [np.float32, float][idx % 2]
                arr = p(np.arange(100 * 101).reshape(1, 100, 101).astype(dtype))
                results.append(percentile(arr, q))
                assert_allclose(results[0], results[-1], type_test=False, atol=1e-4, rtol=1e-4)

    @skip_if_quick
    def test_many_elements_quantile(self):  # pytorch#64947
        for p in TEST_NDARRAYS:
            for elements in (1000, 17_000_000):
                for t in [*TEST_NDARRAYS, list]:
                    x = p(np.random.randn(elements))
                    q = percentile(x, t([10, 50]))
                    if isinstance(x, torch.Tensor):
                        self.assertIsInstance(q, torch.Tensor)
                    assert_allclose(q.shape, [2], type_test=False)

    def test_fails(self):
        for p in TEST_NDARRAYS:
            for q in (-1, 101):
                arr = p(np.arange(100 * 101).reshape(1, 100, 101).astype(np.float32))
                with self.assertRaises(ValueError):
                    percentile(arr, q)

    def test_dim(self):
        q = np.random.randint(0, 100, size=50)
        results = []
        for p in TEST_NDARRAYS:
            arr = p(np.arange(6).reshape(1, 2, 3).astype(np.float32))
            results.append(percentile(arr, q, dim=1))
            assert_allclose(results[0], results[-1], type_test=False, atol=1e-4)

    @parameterized.expand(TEST_MODE)
    def test_mode(self, array, expected, to_long):
        res = mode(array, to_long=to_long)
        assert_allclose(res, expected)

    @parameterized.expand(TEST_MIN_MAX)
    def test_min_max(self, array, input_params, func, expected):
        res = func(array, **input_params)
        assert_allclose(res, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
