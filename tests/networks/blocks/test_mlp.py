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
import torch.nn as nn
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.layers.factories import split_args

TEST_CASE_MLP = []
for dropout_rate in np.linspace(0, 1, 4):
    for hidden_size in [128, 256, 512, 768]:
        for mlp_dim in [0, 1028, 2048, 3072]:
            test_case = [
                {"hidden_size": hidden_size, "mlp_dim": mlp_dim, "dropout_rate": dropout_rate},
                (2, 512, hidden_size),
                (2, 512, hidden_size),
            ]
            TEST_CASE_MLP.append(test_case)

# test different activation layers
TEST_CASE_ACT = []
for act in ["GELU", "GEGLU", ("GEGLU", {})]:  # type: ignore
    TEST_CASE_ACT.append([{"hidden_size": 128, "mlp_dim": 0, "act": act}, (2, 512, 128), (2, 512, 128)])

# test different dropout modes
TEST_CASE_DROP = [["vit", nn.Dropout], ["swin", nn.Dropout], ["vista3d", nn.Identity]]


class TestMLPBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_MLP)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MLPBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            MLPBlock(hidden_size=128, mlp_dim=512, dropout_rate=5.0)

    @parameterized.expand(TEST_CASE_ACT)
    def test_act(self, input_param, input_shape, expected_shape):
        net = MLPBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)
        act_name, _ = split_args(input_param["act"])
        if act_name == "GEGLU":
            self.assertEqual(net.linear1.in_features, net.linear1.out_features // 2)
        else:
            self.assertEqual(net.linear1.in_features, net.linear1.out_features)

    @parameterized.expand(TEST_CASE_DROP)
    def test_dropout_mode(self, dropout_mode, dropout_layer):
        net = MLPBlock(hidden_size=128, mlp_dim=512, dropout_rate=0.1, dropout_mode=dropout_mode)
        self.assertTrue(isinstance(net.drop1, dropout_layer))
        self.assertTrue(isinstance(net.drop2, dropout_layer))


if __name__ == "__main__":
    unittest.main()
