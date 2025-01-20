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
from parameterized import parameterized

from monai.networks.layers import trunc_normal_

TEST_CASES = [
    [{"mean": 0.0, "std": 1.0, "a": 2, "b": 4}, (6, 12, 3, 1, 7)],
    [{"mean": 0.3, "std": 0.6, "a": -1.0, "b": 1.3}, (1, 4, 4, 4)],
    [{"mean": 0.1, "std": 0.4, "a": 1.3, "b": 1.8}, (5, 7, 7, 8, 9)],
]

TEST_ERRORS = [
    [{"mean": 0.0, "std": 1.0, "a": 5, "b": 1.1}, (1, 1, 8, 8, 8)],
    [{"mean": 0.3, "std": -0.1, "a": 1.0, "b": 2.0}, (8, 5, 2, 6, 9)],
    [{"mean": 0.7, "std": 0.0, "a": 0.1, "b": 2.0}, (4, 12, 23, 17)],
]


class TestWeightInit(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape):
        im = torch.rand(input_shape)
        trunc_normal_(im, **input_param)
        self.assertEqual(im.shape, input_shape)

    @parameterized.expand(TEST_ERRORS)
    def test_ill_arg(self, input_param, input_shape):
        with self.assertRaises(ValueError):
            im = torch.rand(input_shape)
            trunc_normal_(im, **input_param)


if __name__ == "__main__":
    unittest.main()
