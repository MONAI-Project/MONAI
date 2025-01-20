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
import torch.nn as nn
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks import SubpixelUpsample
from monai.networks.layers.factories import Conv
from tests.util import SkipIfBeforePyTorchVersion, test_script_save

TEST_CASE_SUBPIXEL = []
for inch in range(1, 5):
    for dim in range(1, 4):
        for factor in range(1, 3):
            test_case = [
                {"spatial_dims": dim, "in_channels": inch, "scale_factor": factor},
                (2, inch, *([8] * dim)),
                (2, inch, *([8 * factor] * dim)),
            ]
            TEST_CASE_SUBPIXEL.append(test_case)

TEST_CASE_SUBPIXEL_2D_EXTRA = [
    {"spatial_dims": 2, "in_channels": 2, "scale_factor": 3},
    (2, 2, 8, 4),  # different size for H and W
    (2, 2, 24, 12),
]

TEST_CASE_SUBPIXEL_3D_EXTRA = [
    {"spatial_dims": 3, "in_channels": 1, "scale_factor": 2},
    (2, 1, 16, 8, 4),  # different size for H, W and D
    (2, 1, 32, 16, 8),
]

conv_block = nn.Sequential(
    Conv[Conv.CONV, 3](1, 4, kernel_size=1), Conv[Conv.CONV, 3](4, 8, kernel_size=3, stride=1, padding=1)
)

TEST_CASE_SUBPIXEL_CONV_BLOCK_EXTRA = [
    {"spatial_dims": 3, "in_channels": 1, "scale_factor": 2, "conv_block": conv_block},
    (2, 1, 16, 8, 4),  # different size for H, W and D
    (2, 1, 32, 16, 8),
]

TEST_CASE_SUBPIXEL.append(TEST_CASE_SUBPIXEL_2D_EXTRA)  # type: ignore
TEST_CASE_SUBPIXEL.append(TEST_CASE_SUBPIXEL_3D_EXTRA)  # type: ignore
TEST_CASE_SUBPIXEL.append(TEST_CASE_SUBPIXEL_CONV_BLOCK_EXTRA)  # type: ignore

# add every test back with the pad/pool sequential component omitted
for tests in list(TEST_CASE_SUBPIXEL):
    args: dict = tests[0]  # type: ignore
    args = dict(args)
    args["apply_pad_pool"] = False
    TEST_CASE_SUBPIXEL.append([args, tests[1], tests[2]])


class TestSUBPIXEL(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SUBPIXEL)
    def test_subpixel_shape(self, input_param, input_shape, expected_shape):
        net = SubpixelUpsample(**input_param)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    @SkipIfBeforePyTorchVersion((1, 8, 1))
    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_SUBPIXEL[0]
        net = SubpixelUpsample(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
