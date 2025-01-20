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

from monai.utils.misc import ensure_tuple
from tests.util import assert_allclose

TESTS = [
    ["test", ("test",)],
    [["test1", "test2"], ("test1", "test2")],
    [123, (123,)],
    [(1, [2], 3), (1, [2], 3)],
    [(1, 2, 3), (1, 2, 3), True],
    [np.array([1, 2]), (np.array([1, 2]),), True],
    [np.array([1, 2]), (1, 2), False],
    [torch.tensor([1, 2]), (torch.tensor([1, 2]),), True],
    [np.array([]), (np.array([]),)],
    [torch.tensor([]), (torch.tensor([]),)],
    [np.array(123), (np.array(123),), True],
    [torch.tensor(123), (torch.tensor(123),)],
]


class TestEnsureTuple(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input, expected_value, wrap_array=False):
        result = ensure_tuple(input, wrap_array)

        self.assertTrue(isinstance(result, tuple))
        if isinstance(input, (np.ndarray, torch.Tensor)):
            for i, j in zip(result, expected_value):
                assert_allclose(i, j)
        else:
            self.assertTupleEqual(result, expected_value)


if __name__ == "__main__":
    unittest.main()
