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
from monai.networks.nets import FullyConnectedNet, VarFullyConnectedNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FC_TEST_CASE_0 = [0]
FC_TEST_CASE_1 = [0.15]

FC_CASES = [FC_TEST_CASE_0, FC_TEST_CASE_1]

VFC_TEST_CASE_0 = [
    {
        "in_channels": 10,
        "out_channels": 10,
        "latent_size": 30,
        "encode_channels": (15, 20, 25),
        "decode_channels": (15, 20, 25),
    },
    (3, 10),
    (3, 10),
]

VFC_CASES = [VFC_TEST_CASE_0]


class TestFullyConnectedNet(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.inSize = 10
        self.arrShape = (self.batch_size, self.inSize)
        self.outSize = 3
        self.channels = [8, 16]
        self.arr = torch.randn(self.arrShape, dtype=torch.float32).to(device)

    @parameterized.expand(FC_CASES)
    def test_fc_shape(self, dropout):
        net = FullyConnectedNet(self.inSize, self.outSize, self.channels, dropout).to(device)
        out = net(self.arr)
        self.assertEqual(out.shape, (self.batch_size, self.outSize))

    @parameterized.expand(VFC_CASES)
    def test_vfc_shape(self, input_param, input_shape, expected_shape):
        net = VarFullyConnectedNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))[0]
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
