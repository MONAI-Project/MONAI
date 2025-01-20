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

from monai.transforms import RandGaussianSmooth
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"sigma_x": (0.5, 1.5), "prob": 1.0},
            p(np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])),
            np.array(
                [
                    [
                        [0.71806467, 0.9074683, 0.71806467],
                        [1.0718315, 1.3545481, 1.0718315],
                        [1.0337002, 1.306359, 1.0337002],
                    ],
                    [
                        [2.0318885, 2.5678391, 2.0318885],
                        [2.6795788, 3.3863702, 2.6795788],
                        [2.3475242, 2.9667296, 2.3475242],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            {"sigma_x": (0.5, 1.5), "sigma_y": (0.5, 1.0), "prob": 1.0},
            p(np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])),
            np.array(
                [
                    [
                        [0.7686928, 0.9848021, 0.7686928],
                        [1.1474025, 1.4699818, 1.1474024],
                        [1.1065826, 1.4176859, 1.1065826],
                    ],
                    [
                        [2.1751494, 2.7866683, 2.1751497],
                        [2.8685062, 3.6749542, 2.8685062],
                        [2.5130394, 3.219552, 2.5130394],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            {"sigma_x": (0.5, 1.5), "sigma_y": (0.5, 1.0), "approx": "scalespace", "prob": 1.0},
            p(np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])),
            np.array(
                [
                    [
                        [0.8128456, 0.96736777, 0.8128456],
                        [1.2742369, 1.5164697, 1.2742369],
                        [1.2800367, 1.5233722, 1.2800368],
                    ],
                    [
                        [2.3825073, 2.8354228, 2.3825073],
                        [3.1855922, 3.7911744, 3.1855922],
                        [2.8496985, 3.391427, 2.8496985],
                    ],
                ]
            ),
        ]
    )


class TestRandGaussianSmooth(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, arguments, image, expected_data):
        converter = RandGaussianSmooth(**arguments)
        converter.set_random_state(seed=0)
        result = converter(image)
        assert_allclose(result, expected_data, rtol=1e-4, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
