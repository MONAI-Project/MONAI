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

from monai.transforms import AddExtremePointsChanneld
from tests.util import TEST_NDARRAYS, assert_allclose

IMG_CHANNEL = 3

TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        TESTS.append(
            [
                {
                    "img": p(np.zeros((IMG_CHANNEL, 4, 3))),
                    "label": q(np.array([[[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]])),
                },
                p(
                    np.array(
                        [
                            [0.38318458, 0.98615628, 0.85551184],
                            [0.35422316, 0.94430935, 1.0],
                            [0.46000731, 0.57319659, 0.46000722],
                            [0.64577687, 0.38318464, 0.0],
                        ]
                    )
                ),
            ]
        )

        TESTS.append(
            [
                {
                    "img": p(np.zeros((IMG_CHANNEL, 4, 3))),
                    "label": q(np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]]])),
                },
                p(
                    np.array(
                        [
                            [0.44628328, 0.80495411, 0.44628328],
                            [0.6779086, 1.0, 0.67790854],
                            [0.33002687, 0.62079221, 0.33002687],
                            [0.0, 0.31848389, 0.0],
                        ]
                    )
                ),
            ]
        )


class TestAddExtremePointsChanneld(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_correct_results(self, input_data, expected):
        add_extreme_points_channel = AddExtremePointsChanneld(
            keys="img", label_key="label", sigma=1.0, rescale_min=0.0, rescale_max=1.0
        )
        result = add_extreme_points_channel(input_data)
        assert_allclose(result["img"][IMG_CHANNEL], expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
