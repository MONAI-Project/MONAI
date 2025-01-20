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

from monai.networks.layers import DropPath

TEST_CASES = [
    [{"drop_prob": 0.0, "scale_by_keep": True}, (1, 8, 8)],
    [{"drop_prob": 0.7, "scale_by_keep": False}, (2, 16, 16, 16)],
    [{"drop_prob": 0.3, "scale_by_keep": True}, (6, 16, 12)],
]

TEST_ERRORS = [[{"drop_prob": 2, "scale_by_keep": False}, (1, 24, 6)]]


class TestDropPath(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape):
        im = torch.rand(input_shape)
        dr_path = DropPath(**input_param)
        out = dr_path(im)
        self.assertEqual(out.shape, input_shape)

    @parameterized.expand(TEST_ERRORS)
    def test_ill_arg(self, input_param, input_shape):
        with self.assertRaises(ValueError):
            DropPath(**input_param)


if __name__ == "__main__":
    unittest.main()
