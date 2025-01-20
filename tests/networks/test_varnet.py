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

from monai.apps.reconstruction.networks.nets.coil_sensitivity_model import CoilSensitivityModel
from monai.apps.reconstruction.networks.nets.complex_unet import ComplexUnet
from monai.apps.reconstruction.networks.nets.varnet import VariationalNetworkModel
from monai.networks import eval_mode
from tests.util import SkipIfBeforePyTorchVersion, test_script_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coil_sens_model = CoilSensitivityModel(spatial_dims=2, features=[8, 16, 32, 64, 128, 8])
refinement_model = ComplexUnet(spatial_dims=2, features=[8, 16, 32, 64, 128, 8])
num_cascades = 2
TESTS = []
TESTS.append([coil_sens_model, refinement_model, num_cascades, (1, 3, 50, 50, 2), (1, 50, 50)])  # batch=1
TESTS.append([coil_sens_model, refinement_model, num_cascades, (2, 3, 50, 50, 2), (2, 50, 50)])  # batch=2


class TestVarNet(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, coil_sens_model, refinement_model, num_cascades, input_shape, expected_shape):
        net = VariationalNetworkModel(coil_sens_model, refinement_model, num_cascades).to(device)
        mask_shape = [1 for _ in input_shape]
        mask_shape[-2] = input_shape[-2]
        mask = torch.zeros(mask_shape)
        mask[..., mask_shape[-2] // 2 - 5 : mask_shape[-2] // 2 + 5, :] = 1

        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device), mask.bool().to(device))
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TESTS)
    @SkipIfBeforePyTorchVersion((1, 9, 1))
    def test_script(self, coil_sens_model, refinement_model, num_cascades, input_shape, expected_shape):
        net = VariationalNetworkModel(coil_sens_model, refinement_model, num_cascades)

        mask_shape = [1 for _ in input_shape]
        mask_shape[-2] = input_shape[-2]
        mask = torch.zeros(mask_shape)
        mask[..., mask_shape[-2] // 2 - 5 : mask_shape[-2] // 2 + 5, :] = 1

        test_data = torch.randn(input_shape)

        test_script_save(net, test_data, mask.bool())


if __name__ == "__main__":
    unittest.main()
