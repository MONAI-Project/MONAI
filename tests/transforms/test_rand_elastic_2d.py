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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Rand2DElastic
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, is_tf32_env

_rtol = 5e-3 if is_tf32_env() else 1e-4

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                {"spacing": (0.3, 0.3), "magnitude_range": (1.0, 2.0), "prob": 0.0, "device": device},
                {"img": p(torch.ones((3, 3, 3))), "spatial_size": (2, 2)},
                p(np.ones((3, 2, 2))),
            ]
        )
        TESTS.append(
            [
                {"spacing": (0.3, 0.3), "magnitude_range": (1.0, 2.0), "prob": 0.0, "device": device},
                {"img": p(torch.arange(27).reshape((3, 3, 3)))},
                p(np.arange(27).reshape((3, 3, 3))),
            ]
        )
        TESTS.append(
            [
                {
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (1.0, 2.0),
                    "prob": 0.9,
                    "device": device,
                    "padding_mode": "zeros",
                },
                {"img": p(torch.ones((3, 3, 3))), "spatial_size": (2, 2), "mode": "bilinear"},
                p(
                    np.array(
                        [
                            [[0.45531988, 0.0], [0.0, 0.71558857]],
                            [[0.45531988, 0.0], [0.0, 0.71558857]],
                            [[0.45531988, 0.0], [0.0, 0.71558857]],
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                {
                    "spacing": (1.0, 1.0),
                    "magnitude_range": (1.0, 1.0),
                    "scale_range": [1.2, 2.2],
                    "prob": 0.9,
                    "padding_mode": "border",
                    "device": device,
                    "spatial_size": (2, 2),
                },
                {"img": p(torch.arange(27).reshape((3, 3, 3)))},
                p(
                    torch.tensor(
                        [
                            [[3.0793, 2.6141], [4.0568, 5.9978]],
                            [[12.0793, 11.6141], [13.0568, 14.9978]],
                            [[21.0793, 20.6141], [22.0568, 23.9978]],
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                {
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (0.1, 0.2),
                    "translate_range": [-0.01, 0.01],
                    "scale_range": [0.01, 0.02],
                    "prob": 0.9,
                    "device": device,
                    "spatial_size": (2, 2),
                },
                {"img": p(torch.arange(27).reshape((3, 3, 3)))},
                p(
                    np.array(
                        [
                            [[1.3584113, 1.9251312], [5.626623, 6.642721]],
                            [[10.358411, 10.925131], [14.626623, 15.642721]],
                            [[19.358412, 19.92513], [23.626623, 24.642721]],
                        ]
                    )
                ),
            ]
        )


class TestRand2DElastic(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_2d_elastic(self, input_param, input_data, expected_val):
        g = Rand2DElastic(**input_param)
        set_track_meta(False)
        result = g(**input_data)
        self.assertNotIsInstance(result, MetaTensor)
        self.assertIsInstance(result, torch.Tensor)
        set_track_meta(True)
        g.set_random_state(123)
        result = g(**input_data)
        assert_allclose(result, expected_val, type_test=False, rtol=_rtol, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
