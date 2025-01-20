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
from monai.networks.nets.vitautoenc import ViTAutoEnc
from tests.util import skip_if_quick, skip_if_windows

TEST_CASE_Vitautoenc = []
for in_channels in [1, 4]:
    for img_size in [64, 96, 128]:
        for patch_size in [16]:
            for proj_type in ["conv", "perceptron"]:
                for nd in [2, 3]:
                    test_case = [
                        {
                            "in_channels": in_channels,
                            "img_size": (img_size,) * nd,
                            "patch_size": (patch_size,) * nd,
                            "hidden_size": 768,
                            "mlp_dim": 3072,
                            "num_layers": 4,
                            "num_heads": 12,
                            "proj_type": proj_type,
                            "dropout_rate": 0.6,
                            "spatial_dims": nd,
                        },
                        (2, in_channels, *([img_size] * nd)),
                        (2, 1, *([img_size] * nd)),
                    ]

                    TEST_CASE_Vitautoenc.append(test_case)

TEST_CASE_Vitautoenc.append(
    [
        {
            "in_channels": 1,
            "img_size": (512, 512, 32),
            "patch_size": (64, 64, 16),
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_layers": 4,
            "num_heads": 12,
            "proj_type": "conv",
            "dropout_rate": 0.6,
            "spatial_dims": 3,
        },
        (2, 1, 512, 512, 32),
        (2, 1, 512, 512, 32),
    ]
)


@skip_if_quick
class TestVitAutoenc(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(4)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @parameterized.expand(TEST_CASE_Vitautoenc)
    @skip_if_windows
    def test_shape(self, input_param, input_shape, expected_shape):
        net = ViTAutoEnc(**input_param)
        with eval_mode(net):
            result, _ = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(
        [
            (1, (32, 32, 32), (64, 64, 64), 512, 3072, 12, 8, "perceptron", 0.3),  # img_size_too_large_for_patch_size
            (1, (96, 96, 96), (8, 8, 8), 512, 3072, 12, 14, "conv", 0.3),  # num_heads_out_of_bound
            (1, (97, 97, 97), (4, 4, 4), 768, 3072, 12, 8, "perceptron", 0.3),  # img_size_not_divisible_by_patch_size
            (4, (96, 96, 96), (16, 16, 16), 768, 3072, 12, 12, "perc", 0.3),  # invalid_pos_embed
            (4, (96, 96, 96), (9, 9, 9), 768, 3072, 12, 12, "perc", 0.3),  # patch_size_not_divisible
            # Add more test cases as needed
        ]
    )
    def test_ill_arg(
        self, in_channels, img_size, patch_size, hidden_size, mlp_dim, num_layers, num_heads, proj_type, dropout_rate
    ):
        with self.assertRaises(ValueError):
            ViTAutoEnc(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                proj_type=proj_type,
                dropout_rate=dropout_rate,
            )


if __name__ == "__main__":
    unittest.main()
