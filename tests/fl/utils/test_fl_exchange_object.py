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

from monai.fl.utils.constants import WeightType
from monai.fl.utils.exchange_object import ExchangeObject
from monai.utils.module import optional_import
from tests.util import SkipIfNoModule

models, has_torchvision = optional_import("torchvision.models")

TEST_INIT_1 = [{"weights": None, "optim": None, "metrics": None, "weight_type": None, "statistics": None}, "{}"]
TEST_INIT_2: list = []
if has_torchvision:
    network = models.resnet18()
    TEST_INIT_2.append(
        {
            "weights": network.state_dict(),
            "optim": torch.optim.Adam(lr=1, params=network.parameters()).state_dict(),
            "metrics": {"accuracy": 1},
            "weight_type": WeightType.WEIGHT_DIFF,
            "statistics": {"some_stat": 1},
        }
    )
    TEST_INIT_2.append("{'weights': 122, 'optim': 2, 'metrics': 1, 'weight_type': fl_weight_diff, 'statistics': 1}")

TEST_FAILURE_METRICS = [{"weights": None, "optim": None, "metrics": 1, "weight_type": None, "statistics": None}]
TEST_FAILURE_STATISTICS = [{"weights": None, "optim": None, "metrics": None, "weight_type": None, "statistics": 1}]
TEST_FAILURE_WEIGHT_TYPE = [{"weights": None, "optim": None, "metrics": None, "weight_type": 1, "statistics": None}]


@SkipIfNoModule("torchvision")
@SkipIfNoModule("ignite")
class TestFLExchangeObject(unittest.TestCase):
    @parameterized.expand([TEST_INIT_1, TEST_INIT_2])
    def test_init(self, input_params, expected_str):
        eo = ExchangeObject(**input_params)
        self.assertIsInstance(eo, ExchangeObject)
        eo.summary()
        self.assertEqual(repr(eo), expected_str)

    @parameterized.expand([TEST_FAILURE_METRICS, TEST_FAILURE_STATISTICS, TEST_FAILURE_WEIGHT_TYPE])
    def test_failures(self, input_params):
        with self.assertRaises(ValueError):
            ExchangeObject(**input_params)


if __name__ == "__main__":
    unittest.main()
