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

from monai.networks.blocks.dints_block import FactorizedReduceBlock

TEST_CASES_3D = [
    [{"in_channel": 32, "out_channel": 16}, (7, 32, 24, 16, 8), (7, 16, 12, 8, 4)],
    [{"in_channel": 16, "out_channel": 32}, (7, 16, 22, 14, 6), (7, 32, 11, 7, 3)],
]


class TestFactRed(unittest.TestCase):

    @parameterized.expand(TEST_CASES_3D)
    def test_factorized_reduce_3d(self, input_param, input_shape, expected_shape):
        net = FactorizedReduceBlock(**input_param)
        result = net(torch.randn(input_shape))
        self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
