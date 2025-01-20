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

from monai.transforms.intensity.dictionary import ComputeHoVerMapsd
from monai.utils import min_version, optional_import
from tests.util import TEST_NDARRAYS, assert_allclose

_, has_skimage = optional_import("skimage", "0.19.0", min_version)

INSTANCE_MASK = np.zeros((1, 16, 16), dtype="int16")
INSTANCE_MASK[:, 5:8, 4:11] = 1
INSTANCE_MASK[:, 3:5, 6:9] = 1
INSTANCE_MASK[:, 8:10, 6:9] = 1
INSTANCE_MASK[:, 13:, 13:] = 2
H_MAP = torch.zeros((16, 16), dtype=torch.float32)
H_MAP[5:8, 4] = -1.0
H_MAP[5:8, 5] = -2.0 / 3.0
H_MAP[3:10, 6] = -1.0 / 3.0
H_MAP[3:10, 7] = 0.0
H_MAP[3:10, 8] = 1.0 / 3.0
H_MAP[5:8, 9] = 2.0 / 3.0
H_MAP[5:8, 10] = 1.0
H_MAP[13:, 13] = -1.0
H_MAP[13:, 14] = 0.0
H_MAP[13:, 15] = 1.0
V_MAP = torch.zeros((16, 16), dtype=torch.float32)
V_MAP[3, 6:9] = -1.0
V_MAP[4, 6:9] = -2.0 / 3.0
V_MAP[5, 4:11] = -1.0 / 3.0
V_MAP[6, 4:11] = 0.0
V_MAP[7, 4:11] = 1.0 / 3.0
V_MAP[8, 6:9] = 2.0 / 3.0
V_MAP[9, 6:9] = 1.0
V_MAP[13, 13:] = -1.0
V_MAP[14, 13:] = 0.0
V_MAP[15, 13:] = 1.0
HV_MAPS = torch.stack([H_MAP, V_MAP])
TEST_CASE_0 = [{}, {"mask": INSTANCE_MASK}, {"hover_mask": HV_MAPS}]
TEST_CASE_1 = [{"dtype": "float64"}, {"mask": INSTANCE_MASK}, {"hover_mask": HV_MAPS}]
TEST_CASE_1 = [{"dtype": "float64", "new_key_prefix": ""}, {"mask": INSTANCE_MASK}, {"mask": HV_MAPS}]

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append([p, *TEST_CASE_0])
    TESTS.append([p, *TEST_CASE_1])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class ComputeHoVerMapsDictTests(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_horizontal_certical_maps(self, in_type, arguments, mask, hv_mask):
        hv_key = list(hv_mask.keys())[0]
        input_image = {}
        for k in mask.keys():
            input_image[k] = in_type(mask[k])
        result = ComputeHoVerMapsd(keys="mask", **arguments)(input_image)[hv_key]
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(str(result.dtype).split(".")[1], arguments.get("dtype", "float32"))
        assert_allclose(result, hv_mask[hv_key], type_test="tensor")


if __name__ == "__main__":
    unittest.main()
