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
from parameterized import parameterized

from monai.transforms import HistogramNormalized
from monai.utils import get_equivalent_dtype
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"keys": "img", "num_bins": 4, "min": 1, "max": 5, "mask_key": "mask"},
            {"img": p(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])), "mask": p(np.array([1, 1, 1, 1, 1, 0]))},
            p(np.array([1.0, 1.5, 2.5, 4.0, 5.0, 5.0])),
        ]
    )

    TESTS.append(
        [
            {"keys": "img", "num_bins": 4, "max": 4, "dtype": np.uint8},
            {"img": p(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))},
            p(np.array([0, 0, 1, 3, 4])),
        ]
    )

    TESTS.append(
        [
            {"keys": "img", "num_bins": 256, "max": 255, "dtype": np.uint8},
            {"img": p(np.array([[[100.0, 200.0], [150.0, 250.0]]]))},
            p(np.array([[[0, 170], [70, 255]]])),
        ]
    )


class TestHistogramNormalized(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, arguments, image, expected_data):
        result = HistogramNormalized(**arguments)(image)["img"]
        assert_allclose(result, expected_data, type_test="tensor")
        self.assertEqual(get_equivalent_dtype(result.dtype, data_type=np.ndarray), arguments.get("dtype", np.float32))


if __name__ == "__main__":
    unittest.main()
