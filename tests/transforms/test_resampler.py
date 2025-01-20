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

from monai.transforms import Resample
from monai.transforms.utils import create_grid
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for q in TEST_NDARRAYS_ALL:
        for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {"grid": p(create_grid((2, 2))), "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(np.array([[[0.0, 1.0], [2.0, 3.0]]])),
                ]
            )
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {"grid": p(create_grid((4, 4))), "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(
                        np.array(
                            [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
                        )
                    ),
                ]
            )
            TESTS.append(
                [
                    dict(padding_mode="border", device=device),
                    {"grid": p(create_grid((4, 4))), "img": q(np.arange(4).reshape((1, 2, 2)))},
                    q(
                        np.array(
                            [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3, 3.0], [2.0, 2.0, 3.0, 3.0]]]
                        )
                    ),
                ]
            )
            # TESTS.append(  # not well defined nearest + reflection resampling
            #     [
            #         dict(padding_mode="reflection", device=device),
            #         {"grid": p(create_grid((4, 4))), "img": q(np.arange(4).reshape((1, 2, 2))), "mode": "nearest"},
            #         q(
            #             np.array(
            #                 [[[3.0, 2.0, 3.0, 2.0], [1.0, 0.0, 1.0, 0.0], [3.0, 2.0, 3.0, 2.0], [1.0, 0.0, 1.0, 0.0]]]
            #             )
            #         ),
            #     ]
            # )
            TESTS.append(
                [
                    dict(padding_mode="zeros", device=device),
                    {
                        "grid": p(create_grid((4, 4, 4))),
                        "img": q(np.arange(8).reshape((1, 2, 2, 2))),
                        "mode": "bilinear",
                    },
                    q(
                        np.array(
                            [
                                [
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 2.0, 3.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 4.0, 5.0, 0.0],
                                        [0.0, 6.0, 7.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                    [
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0],
                                    ],
                                ]
                            ]
                        )
                    ),
                ]
            )
            TESTS.append(
                [
                    dict(padding_mode="border", device=device),
                    {
                        "grid": p(create_grid((4, 4, 4))),
                        "img": q(np.arange(8).reshape((1, 2, 2, 2))),
                        "mode": "bilinear",
                    },
                    q(
                        np.array(
                            [
                                [
                                    [
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                    ],
                                    [
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                        [2.0, 2.0, 3.0, 3.0],
                                    ],
                                    [
                                        [4.0, 4.0, 5.0, 5.0],
                                        [4.0, 4.0, 5.0, 5.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                    ],
                                    [
                                        [4.0, 4.0, 5.0, 5.0],
                                        [4.0, 4.0, 5.0, 5.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                        [6.0, 6.0, 7.0, 7.0],
                                    ],
                                ]
                            ]
                        )
                    ),
                ]
            )


class TestResample(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_resample(self, input_param, input_data, expected_val):
        g = Resample(**input_param)
        result = g(**input_data)
        if "device" in input_data:
            self.assertEqual(result.device, input_data["device"])
        assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
