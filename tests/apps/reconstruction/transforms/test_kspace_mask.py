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
from parameterized import parameterized

from monai.apps.reconstruction.transforms.array import EquispacedKspaceMask, RandomKspaceMask
from monai.utils.type_conversion import convert_data_type

# test case for apply_mask
ksp, *_ = convert_data_type(np.ones([50, 50, 2]), torch.Tensor)
TESTSM = [(ksp,)]


class TestMRIUtils(unittest.TestCase):

    @parameterized.expand(TESTSM)
    def test_mask(self, test_data):
        # random mask
        masker = RandomKspaceMask(center_fractions=[0.08], accelerations=[4.0], spatial_dims=1, is_complex=True)
        masker.set_random_state(seed=0)
        result, _ = masker(test_data)
        mask = masker.mask
        result = result[..., mask.squeeze() == 0, :].sum()
        self.assertEqual(result.item(), 0)

        # equispaced mask
        masker = EquispacedKspaceMask(center_fractions=[0.08], accelerations=[4.0], spatial_dims=1, is_complex=True)
        masker.set_random_state(seed=0)
        result, _ = masker(test_data)
        mask = masker.mask
        result = result[..., mask.squeeze() == 0, :].sum()
        self.assertEqual(result.item(), 0)


if __name__ == "__main__":
    unittest.main()
