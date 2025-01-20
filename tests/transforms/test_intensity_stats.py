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

from monai.transforms import IntensityStats
from tests.util import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.extend(
        [
            [
                {"ops": ["max", "mean"], "key_prefix": "orig"},
                p([[[0.0, 1.0], [2.0, 3.0]]]),
                {"affine": None},
                {"orig_max": 3.0, "orig_mean": 1.5},
            ],
            [{"ops": "std", "key_prefix": "orig"}, p([[[0.0, 1.0], [2.0, 3.0]]]), None, {"orig_std": 1.118034}],
            [
                {"ops": [np.mean, "max", np.min], "key_prefix": "orig"},
                p([[[0.0, 1.0], [2.0, 3.0]]]),
                None,
                {"orig_custom_0": 1.5, "orig_max": 3.0, "orig_custom_1": 0.0},
            ],
            [
                {"ops": ["max", "mean"], "key_prefix": "orig", "channel_wise": True},
                p([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]),
                {"affine": None},
                {"orig_max": [3.0, 7.0], "orig_mean": [1.5, 5.5]},
            ],
            [
                {"ops": ["max", "mean"], "key_prefix": "orig"},
                p([[[0.0, 1.0], [2.0, 3.0]]]),
                {"affine": None},
                {"orig_max": 3.0, "orig_mean": 1.5},
            ],
        ]
    )


class TestIntensityStats(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_param, img, meta_dict, expected):
        _, meta_dict = IntensityStats(**input_param)(img, meta_dict)
        for k, v in expected.items():
            self.assertTrue(k in meta_dict)
            np.testing.assert_allclose(v, meta_dict[k], atol=1e-3)

    def test_mask(self):
        for p in TEST_NDARRAYS:
            img = p([[[0.0, 1.0], [2.0, 3.0]]])
            mask = np.array([[[1, 0], [1, 0]]], dtype=bool)
            img, meta_dict = IntensityStats(ops=["max", "mean"], key_prefix="orig")(img, mask=mask)
            np.testing.assert_allclose(meta_dict["orig_max"], 2.0, atol=1e-3)
            np.testing.assert_allclose(meta_dict["orig_mean"], 1.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
