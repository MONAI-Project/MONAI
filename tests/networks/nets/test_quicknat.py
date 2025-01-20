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
from monai.networks.nets import Quicknat
from monai.utils import optional_import
from tests.util import test_script_save

_, has_se = optional_import("squeeze_and_excitation")

TEST_CASES = [
    # params, input_shape, expected_shape
    [{"num_classes": 1, "num_channels": 1, "num_filters": 1, "se_block": None}, (1, 1, 32, 32), (1, 1, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 4, "se_block": None}, (1, 1, 64, 64), (1, 1, 64, 64)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": None}, (1, 1, 128, 128), (1, 1, 128, 128)],
    [{"num_classes": 4, "num_channels": 1, "num_filters": 64, "se_block": None}, (1, 1, 32, 32), (1, 4, 32, 32)],
    [{"num_classes": 33, "num_channels": 1, "num_filters": 64, "se_block": None}, (1, 1, 32, 32), (1, 33, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": "CSE"}, (1, 1, 32, 32), (1, 1, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": "SSE"}, (1, 1, 32, 32), (1, 1, 32, 32)],
    [{"num_classes": 1, "num_channels": 1, "num_filters": 64, "se_block": "CSSE"}, (1, 1, 32, 32), (1, 1, 32, 32)],
]


@unittest.skipUnless(has_se, "squeeze_and_excitation not installed")
class TestQuicknat(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = Quicknat(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = Quicknat(num_classes=1, num_channels=1)
        test_data = torch.randn(16, 1, 32, 32)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
