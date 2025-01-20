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
from monai.networks.blocks import SEBlock
from monai.networks.layers.factories import Act, Norm
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    [
        {"spatial_dims": 2, "in_channels": 4, "n_chns_1": 20, "n_chns_2": 30, "n_chns_3": 4, "r": 2},
        (7, 4, 64, 48),  # 4-channel 2D, batch 7
        (7, 4, 64, 48),
    ],
    [
        {"spatial_dims": 1, "in_channels": 3, "n_chns_1": 20, "n_chns_2": 30, "n_chns_3": 40, "r": 5},
        (16, 3, 63),  # 3-channel 1D, batch 16
        (16, 40, 63),
    ],
]

TEST_CASES_3D = []
for type_1 in (
    {"kernel_size": 3, "act": Act.PRELU, "norm": Norm.INSTANCE},
    {"kernel_size": 1, "act": None, "norm": Norm.INSTANCE},
):
    for type_2 in (
        {"kernel_size": 3, "act": Act.PRELU, "norm": Norm.INSTANCE},
        {"kernel_size": 1, "act": None, "norm": Norm.INSTANCE},
    ):
        test_case = [
            {
                "spatial_dims": 3,
                "in_channels": 10,
                "r": 3,
                "n_chns_1": 3,
                "n_chns_2": 5,
                "n_chns_3": 11,
                "conv_param_1": type_1,
                "conv_param_3": type_2,
            },
            (16, 10, 32, 24, 48),  # 10-channel 3D, batch 16
            (16, 11, 32, 24, 48),
        ]
        TEST_CASES_3D.append(test_case)


class TestSEBlockLayer(unittest.TestCase):
    @parameterized.expand(TEST_CASES + TEST_CASES_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SEBlock(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_CASES[0]
        net = SEBlock(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SEBlock(spatial_dims=1, in_channels=4, n_chns_1=2, n_chns_2=3, n_chns_3=4, r=100)


if __name__ == "__main__":
    unittest.main()
