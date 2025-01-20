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

from monai.transforms import RandAffineGrid
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, is_tf32_env

_rtol = 1e-1 if is_tf32_env() else 1e-4

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append([{"device": device}, {"grid": p(torch.ones((3, 3, 3)))}, p(np.ones((3, 3, 3)))])
        TESTS.append(
            [
                {"rotate_range": (1, 2), "translate_range": (3, 3, 3)},
                {"grid": p(torch.arange(0, 27).reshape((3, 3, 3)))},
                p(
                    np.array(
                        [
                            [
                                [-32.81998, -33.910976, -35.001972],
                                [-36.092968, -37.183964, -38.27496],
                                [-39.36596, -40.456955, -41.54795],
                            ],
                            [
                                [2.1380205, 3.1015975, 4.0651755],
                                [5.028752, 5.9923296, 6.955907],
                                [7.919484, 8.883063, 9.84664],
                            ],
                            [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                {"translate_range": (3, 3, 3), "device": device},
                {"spatial_size": (3, 3, 3)},
                np.array(
                    [
                        [
                            [
                                [0.17881513, 0.17881513, 0.17881513],
                                [0.17881513, 0.17881513, 0.17881513],
                                [0.17881513, 0.17881513, 0.17881513],
                            ],
                            [
                                [1.1788151, 1.1788151, 1.1788151],
                                [1.1788151, 1.1788151, 1.1788151],
                                [1.1788151, 1.1788151, 1.1788151],
                            ],
                            [
                                [2.1788151, 2.1788151, 2.1788151],
                                [2.1788151, 2.1788151, 2.1788151],
                                [2.1788151, 2.1788151, 2.1788151],
                            ],
                        ],
                        [
                            [
                                [-2.283164, -2.283164, -2.283164],
                                [-1.283164, -1.283164, -1.283164],
                                [-0.28316402, -0.28316402, -0.28316402],
                            ],
                            [
                                [-2.283164, -2.283164, -2.283164],
                                [-1.283164, -1.283164, -1.283164],
                                [-0.28316402, -0.28316402, -0.28316402],
                            ],
                            [
                                [-2.283164, -2.283164, -2.283164],
                                [-1.283164, -1.283164, -1.283164],
                                [-0.28316402, -0.28316402, -0.28316402],
                            ],
                        ],
                        [
                            [
                                [-2.6388912, -1.6388912, -0.6388912],
                                [-2.6388912, -1.6388912, -0.6388912],
                                [-2.6388912, -1.6388912, -0.6388912],
                            ],
                            [
                                [-2.6388912, -1.6388912, -0.6388912],
                                [-2.6388912, -1.6388912, -0.6388912],
                                [-2.6388912, -1.6388912, -0.6388912],
                            ],
                            [
                                [-2.6388912, -1.6388912, -0.6388912],
                                [-2.6388912, -1.6388912, -0.6388912],
                                [-2.6388912, -1.6388912, -0.6388912],
                            ],
                        ],
                        [
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        ],
                    ]
                ),
            ]
        )
        TESTS.append(
            [
                {"device": device, "rotate_range": (1.0, 1.0, 1.0), "shear_range": (0.1,), "scale_range": (1.2,)},
                {"grid": p(torch.arange(0, 108).reshape((4, 3, 3, 3)))},
                p(
                    np.array(
                        [
                            [
                                [
                                    [-9.4201e00, -8.1672e00, -6.9143e00],
                                    [-5.6614e00, -4.4085e00, -3.1556e00],
                                    [-1.9027e00, -6.4980e-01, 6.0310e-01],
                                ],
                                [
                                    [1.8560e00, 3.1089e00, 4.3618e00],
                                    [5.6147e00, 6.8676e00, 8.1205e00],
                                    [9.3734e00, 1.0626e01, 1.1879e01],
                                ],
                                [
                                    [1.3132e01, 1.4385e01, 1.5638e01],
                                    [1.6891e01, 1.8144e01, 1.9397e01],
                                    [2.0650e01, 2.1902e01, 2.3155e01],
                                ],
                            ],
                            [
                                [
                                    [9.9383e-02, -4.8845e-01, -1.0763e00],
                                    [-1.6641e00, -2.2519e00, -2.8398e00],
                                    [-3.4276e00, -4.0154e00, -4.6032e00],
                                ],
                                [
                                    [-5.1911e00, -5.7789e00, -6.3667e00],
                                    [-6.9546e00, -7.5424e00, -8.1302e00],
                                    [-8.7180e00, -9.3059e00, -9.8937e00],
                                ],
                                [
                                    [-1.0482e01, -1.1069e01, -1.1657e01],
                                    [-1.2245e01, -1.2833e01, -1.3421e01],
                                    [-1.4009e01, -1.4596e01, -1.5184e01],
                                ],
                            ],
                            [
                                [
                                    [5.9635e01, 6.1199e01, 6.2764e01],
                                    [6.4328e01, 6.5892e01, 6.7456e01],
                                    [6.9021e01, 7.0585e01, 7.2149e01],
                                ],
                                [
                                    [7.3714e01, 7.5278e01, 7.6842e01],
                                    [7.8407e01, 7.9971e01, 8.1535e01],
                                    [8.3099e01, 8.4664e01, 8.6228e01],
                                ],
                                [
                                    [8.7792e01, 8.9357e01, 9.0921e01],
                                    [9.2485e01, 9.4049e01, 9.5614e01],
                                    [9.7178e01, 9.8742e01, 1.0031e02],
                                ],
                            ],
                            [
                                [
                                    [8.1000e01, 8.2000e01, 8.3000e01],
                                    [8.4000e01, 8.5000e01, 8.6000e01],
                                    [8.7000e01, 8.8000e01, 8.9000e01],
                                ],
                                [
                                    [9.0000e01, 9.1000e01, 9.2000e01],
                                    [9.3000e01, 9.4000e01, 9.5000e01],
                                    [9.6000e01, 9.7000e01, 9.8000e01],
                                ],
                                [
                                    [9.9000e01, 1.0000e02, 1.0100e02],
                                    [1.0200e02, 1.0300e02, 1.0400e02],
                                    [1.0500e02, 1.0600e02, 1.0700e02],
                                ],
                            ],
                        ]
                    )
                ),
            ]
        )


class TestRandAffineGrid(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_affine_grid(self, input_param, input_data, expected_val):
        g = RandAffineGrid(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        if "device" in input_data:
            self.assertEqual(result.device, input_data[device])
        assert_allclose(result, expected_val, type_test=False, rtol=_rtol, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
