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
from copy import deepcopy

from parameterized import parameterized

from monai.transforms import generate_label_classes_crop_centers
from monai.utils.misc import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose

TEST_CASE_1 = [
    {
        "spatial_size": [2, 2, 2],
        "num_samples": 2,
        "ratios": [1, 2],
        "label_spatial_shape": [3, 3, 3],
        "indices": [[3, 12, 21], [1, 9, 18]],
    },
    tuple,
    2,
    3,
]

TEST_CASE_2 = [
    {
        "spatial_size": [2, 2, 2],
        "num_samples": 1,
        "ratios": None,
        "label_spatial_shape": [3, 3, 3],
        "indices": [[3, 12, 21], [1, 9, 18]],
    },
    tuple,
    1,
    3,
]


class TestGenerateLabelClassesCropCenters(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_data, expected_type, expected_count, expected_shape):
        results = []
        for p in TEST_NDARRAYS + (None,):
            input_data = deepcopy(input_data)
            if p is not None:
                input_data["indices"] = p(input_data["indices"])
            set_determinism(0)
            result = generate_label_classes_crop_centers(**input_data)
            self.assertIsInstance(result, expected_type)
            self.assertEqual(len(result), expected_count)
            self.assertEqual(len(result[0]), expected_shape)
            # check for consistency between numpy, torch and torch.cuda
            results.append(result)
            if len(results) > 1:
                for x, y in zip(result[0], result[-1]):
                    assert_allclose(x, y, type_test=False)


if __name__ == "__main__":
    unittest.main()
