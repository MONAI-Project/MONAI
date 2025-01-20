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
from monai.networks.nets import NetAdapter, resnet18
from tests.util import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [{"num_classes": 1, "use_conv": True, "dim": 2}, (2, 3, 224, 224), (2, 1, 8, 1)]

TEST_CASE_1 = [{"num_classes": 1, "use_conv": True, "dim": 3, "pool": None}, (2, 3, 32, 32, 32), (2, 1, 1, 1, 1)]

TEST_CASE_2 = [{"num_classes": 5, "use_conv": True, "dim": 3, "pool": None}, (2, 3, 32, 32, 32), (2, 5, 1, 1, 1)]

TEST_CASE_3 = [
    {"num_classes": 5, "use_conv": True, "pool": ("avg", {"kernel_size": 4, "stride": 1}), "dim": 3},
    (2, 3, 128, 128, 128),
    (2, 5, 5, 1, 1),
]

TEST_CASE_4 = [
    {"num_classes": 5, "use_conv": False, "pool": ("adaptiveavg", {"output_size": (1, 1, 1)}), "dim": 3},
    (2, 3, 32, 32, 32),
    (2, 5),
]


class TestNetAdapter(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_shape(self, input_param, input_shape, expected_shape):
        spatial_dims = input_param["dim"]
        stride = (1, 2, 2)[:spatial_dims]
        model = resnet18(spatial_dims=spatial_dims, conv1_t_stride=stride)
        input_param["model"] = model
        net = NetAdapter(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_0])
    def test_script(self, input_param, input_shape, expected_shape):
        spatial_dims = input_param["dim"]
        stride = (1, 2, 2)[:spatial_dims]
        model = resnet18(spatial_dims=spatial_dims, conv1_t_stride=stride)
        input_param["model"] = model
        net = NetAdapter(**input_param).to("cpu")
        test_data = torch.randn(input_shape).to("cpu")
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
