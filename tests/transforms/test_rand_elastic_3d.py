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
from monai.transforms import Rand3DElastic
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                {
                    "magnitude_range": (0.3, 2.3),
                    "sigma_range": (1.0, 20.0),
                    "prob": 0.0,
                    "device": device,
                    "spatial_size": -1,
                },
                {"img": p(torch.arange(72).reshape((2, 3, 3, 4)))},
                p(np.arange(72).reshape((2, 3, 3, 4))),
            ]
        )
        TESTS.append(
            [
                {"magnitude_range": (0.3, 2.3), "sigma_range": (1.0, 20.0), "prob": 0.0, "device": device},
                {"img": p(torch.ones((2, 3, 3, 3))), "spatial_size": (2, 2, 2)},
                p(np.ones((2, 2, 2, 2))),
            ]
        )
        TESTS.append(
            [
                {"magnitude_range": (0.3, 0.3), "sigma_range": (1.0, 2.0), "prob": 0.9, "device": device},
                {"img": p(torch.arange(27).reshape((1, 3, 3, 3))), "spatial_size": (2, 2, 2)},
                p(
                    np.array(
                        [
                            [
                                [[6.4939356, 7.50289], [9.518351, 10.522849]],
                                [[15.512375, 16.523542], [18.531467, 19.53646]],
                            ]
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                {
                    "magnitude_range": (0.3, 0.3),
                    "sigma_range": (1.0, 2.0),
                    "prob": 0.9,
                    "rotate_range": [1, 1, 1],
                    "device": device,
                    "spatial_size": (2, 2, 2),
                },
                {"img": p(torch.arange(27).reshape((1, 3, 3, 3))), "mode": "bilinear"},
                p(
                    np.array(
                        [
                            [
                                [[5.0069294, 9.463932], [9.287769, 13.739735]],
                                [[12.319424, 16.777205], [16.594296, 21.045748]],
                            ]
                        ]
                    )
                ),
            ]
        )


class TestRand3DElastic(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_3d_elastic(self, input_param, input_data, expected_val):
        g = Rand3DElastic(**input_param)
        set_track_meta(False)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertNotIsInstance(result, MetaTensor)
        self.assertIsInstance(result, torch.Tensor)
        set_track_meta(True)
        g.set_random_state(123)
        result = g(**input_data)
        assert_allclose(result, expected_val, type_test=False, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
