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

from monai.networks import eval_mode
from monai.networks.blocks import UpSample
from monai.utils import UpsampleMode

TEST_CASES = [
    [{"spatial_dims": 2, "in_channels": 4}, (7, 4, 32, 48), (7, 4, 64, 96)],  # 4-channel 2D, batch 7
    [{"spatial_dims": 1, "in_channels": 4, "out_channels": 3}, (16, 4, 63), (16, 3, 126)],  # 4-channel 1D, batch 16
    [
        {"spatial_dims": 1, "in_channels": 4, "out_channels": 8, "mode": "deconv", "align_corners": False},
        (16, 4, 20),
        (16, 8, 40),
    ],  # 4-channel 1D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 4, "mode": "nontrainable"},
        (16, 4, 32, 24, 48),
        (16, 4, 64, 48, 96),
    ],  # 4-channel 3D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 4, "mode": "nontrainable", "size": 64},
        (16, 4, 32, 24, 48),
        (16, 4, 64, 64, 64),
    ],  # 4-channel 3D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 4, "mode": "nontrainable", "size": (64, 24, 48)},
        (16, 4, 32, 24, 48),
        (16, 4, 64, 24, 48),
    ],  # 4-channel 3D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 1, "mode": "deconv", "scale_factor": 3, "align_corners": False},
        (16, 1, 10, 15, 20),
        (16, 1, 30, 45, 60),
    ],  # 1-channel 3D, batch 16
    [
        {"spatial_dims": 3, "in_channels": 1, "mode": "pixelshuffle", "scale_factor": 2, "align_corners": False},
        (16, 1, 10, 15, 20),
        (16, 1, 20, 30, 40),
    ],  # 1-channel 3D, batch 16
    [
        {"spatial_dims": 2, "in_channels": 4, "mode": "pixelshuffle", "scale_factor": 2},
        (16, 4, 10, 15),
        (16, 4, 20, 30),
    ],  # 4-channel 2D, batch 16
    [
        {
            "spatial_dims": 3,
            "mode": "pixelshuffle",
            "scale_factor": 2,
            "align_corners": False,
            "pre_conv": torch.nn.Conv3d(in_channels=1, out_channels=24, kernel_size=3, stride=1, padding=1),
        },
        (16, 1, 10, 15, 20),
        (16, 3, 20, 30, 40),
    ],  # 1-channel 3D, batch 16, pre_conv
    [
        {"spatial_dims": 3, "in_channels": 8, "out_channels": 4, "mode": "deconvgroup"},
        (16, 8, 16, 16, 16),
        (16, 4, 32, 32, 32),
    ],  # 8-channel 3D, batch 16
    [
        {"spatial_dims": 2, "in_channels": 32, "out_channels": 16, "mode": "deconvgroup", "scale_factor": 2},
        (8, 32, 16, 16),
        (8, 16, 32, 32),
    ],  # 32-channel 2D, batch 8
]

TEST_CASES_EQ = []
for s in range(1, 5):
    expected_shape = (16, 5, 4 * s, 5 * s, 6 * s)
    for t in UpsampleMode:
        test_case = [
            {
                "spatial_dims": 3,
                "in_channels": 3,
                "out_channels": 5,
                "mode": t,
                "scale_factor": s,
                "align_corners": True,
            },
            (16, 3, 4, 5, 6),
            expected_shape,
        ]
        TEST_CASES_EQ.append(test_case)

TEST_CASES_EQ2 = []  # type: ignore
for s in range(2, 5):
    for k in range(1, 7):
        expected_shape = (16, 5, 4 * s, 5 * s, 6 * s)
        for t in UpsampleMode:
            test_case = [
                {
                    "spatial_dims": 3,
                    "in_channels": 3,
                    "out_channels": 5,
                    "mode": t,
                    "scale_factor": s,
                    "kernel_size": k,
                    "align_corners": False,
                },
                (16, 3, 4, 5, 6),
                expected_shape,
            ]
            TEST_CASES_EQ.append(test_case)


class TestUpsample(unittest.TestCase):

    @parameterized.expand(TEST_CASES + TEST_CASES_EQ + TEST_CASES_EQ2)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = UpSample(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape, msg=str(input_param))


if __name__ == "__main__":
    unittest.main()
