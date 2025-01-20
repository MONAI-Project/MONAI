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
from monai.networks.blocks import SimpleASPP

TEST_CASES = [
    [  # 32-channel 2D, batch 7
        {"spatial_dims": 2, "in_channels": 32, "conv_out_channels": 3, "norm_type": ("batch", {"affine": False})},
        (7, 32, 18, 20),
        (7, 12, 18, 20),
    ],
    [  # 4-channel 1D, batch 16
        {"spatial_dims": 1, "in_channels": 4, "conv_out_channels": 8, "acti_type": ("PRELU", {"num_parameters": 32})},
        (16, 4, 17),
        (16, 32, 17),
    ],
    [  # 3-channel 3D, batch 16
        {"spatial_dims": 3, "in_channels": 3, "conv_out_channels": 2},
        (16, 3, 17, 18, 19),
        (16, 8, 17, 18, 19),
    ],
    [  # 3-channel 3D, batch 16
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "conv_out_channels": 2,
            "kernel_sizes": (1, 3, 3),
            "dilations": (1, 2, 4),
        },
        (16, 3, 17, 18, 19),
        (16, 6, 17, 18, 19),
    ],
]

TEST_ILL_CASES = [
    [  # 3-channel 3D, batch 16, wrong k and d sizes.
        {"spatial_dims": 3, "in_channels": 3, "conv_out_channels": 2, "kernel_sizes": (1, 3, 3), "dilations": (1, 2)},
        (16, 3, 17, 18, 19),
        ValueError,
    ],
    [  # 3-channel 3D, batch 16, wrong k and d sizes.
        {
            "spatial_dims": 3,
            "in_channels": 3,
            "conv_out_channels": 2,
            "kernel_sizes": (1, 3, 4),
            "dilations": (1, 2, 3),
        },
        (16, 3, 17, 18, 19),
        NotImplementedError,  # unknown padding k=4, d=3
    ],
]


class TestChannelSELayer(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SimpleASPP(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_ILL_CASES)
    def test_ill_args(self, input_param, input_shape, error_type):
        with self.assertRaises(error_type):
            SimpleASPP(**input_param)


if __name__ == "__main__":
    unittest.main()
