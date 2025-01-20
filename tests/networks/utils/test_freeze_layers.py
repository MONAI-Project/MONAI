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

from monai.networks.utils import freeze_layers
from monai.utils import set_determinism
from tests.networks.utils.test_copy_model_state import _TestModelOne, _TestModelTwo

TEST_CASES = []
__devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
for _x in __devices:
    TEST_CASES.append(_x)


class TestModuleState(unittest.TestCase):
    def tearDown(self):
        set_determinism(None)

    @parameterized.expand(TEST_CASES)
    def test_freeze_vars(self, device):
        set_determinism(0)
        model = _TestModelOne(10, 20, 3)
        model.to(device)
        freeze_layers(model, "class")

        for name, param in model.named_parameters():
            if "class_layer" in name:
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)

    @parameterized.expand(TEST_CASES)
    def test_exclude_vars(self, device):
        set_determinism(0)
        model = _TestModelTwo(10, 20, 10, 4)
        model.to(device)
        freeze_layers(model, exclude_vars="class")

        for name, param in model.named_parameters():
            if "class_layer" in name:
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
