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

from parameterized import parameterized

from monai.transforms import GaussianSharpen
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []

for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {},
            p([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
            p(
                [
                    [
                        [4.1081963, 3.4950666, 4.1081963],
                        [3.7239995, 2.8491793, 3.7239995],
                        [4.569839, 3.9529324, 4.569839],
                    ],
                    [[10.616725, 9.081067, 10.616725], [9.309998, 7.12295, 9.309998], [11.078365, 9.538931, 11.078365]],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            {"sigma1": 1.0, "sigma2": 0.75, "alpha": 20},
            p([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
            p(
                [
                    [
                        [4.513644, 4.869134, 4.513644],
                        [8.467242, 9.4004135, 8.467242],
                        [10.416813, 12.0653515, 10.416813],
                    ],
                    [
                        [15.711488, 17.569994, 15.711488],
                        [21.16811, 23.501041, 21.16811],
                        [21.614658, 24.766209, 21.614658],
                    ],
                ]
            ),
        ]
    )

    TESTS.append(
        [
            {"sigma1": (0.5, 1.0), "sigma2": (0.5, 0.75), "alpha": 20},
            p([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]),
            p(
                [
                    [
                        [3.3324685, 3.335536, 3.3324673],
                        [7.7666636, 8.16056, 7.7666636],
                        [12.662973, 14.317837, 12.6629715],
                    ],
                    [
                        [15.329051, 16.57557, 15.329051],
                        [19.41665, 20.40139, 19.416655],
                        [24.659554, 27.557873, 24.659554],
                    ],
                ]
            ),
        ]
    )


class TestGaussianSharpen(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, arguments, image, expected_data):
        result = GaussianSharpen(**arguments)(image)
        assert_allclose(result, expected_data, atol=0, rtol=1e-4, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
