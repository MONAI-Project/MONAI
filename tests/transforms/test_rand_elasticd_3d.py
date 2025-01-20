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

from monai.transforms import Rand3DElasticd
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "magnitude_range": (0.3, 2.3),
                    "sigma_range": (1.0, 20.0),
                    "prob": 0.0,
                    "device": device,
                    "spatial_size": (2, 2, 2),
                },
                {"img": p(torch.ones((2, 3, 3, 3))), "seg": p(torch.ones((2, 3, 3, 3)))},
                p(np.ones((2, 2, 2, 2))),
            ]
        )
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "magnitude_range": (0.3, 2.3),
                    "sigma_range": (1.0, 20.0),
                    "prob": 0.0,
                    "device": device,
                    "spatial_size": (2, -1, -1),
                },
                {"img": p(torch.ones((2, 3, 3, 3))), "seg": p(torch.ones((2, 3, 3, 3)))},
                p(np.ones((2, 2, 3, 3))),
            ]
        )
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "magnitude_range": (0.3, 2.3),
                    "sigma_range": (1.0, 20.0),
                    "prob": 0.0,
                    "device": device,
                    "spatial_size": -1,
                },
                {"img": p(torch.arange(8).reshape((1, 2, 2, 2))), "seg": p(torch.arange(8).reshape((1, 2, 2, 2)))},
                p(np.arange(8).reshape((1, 2, 2, 2))),
            ]
        )
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "magnitude_range": (0.3, 0.3),
                    "sigma_range": (1.0, 2.0),
                    "prob": 0.9,
                    "device": device,
                    "spatial_size": (2, 2, 2),
                },
                {"img": p(torch.arange(27).reshape((1, 3, 3, 3))), "seg": p(torch.arange(27).reshape((1, 3, 3, 3)))},
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
                    "keys": ("img", "seg"),
                    "magnitude_range": (0.3, 0.3),
                    "sigma_range": (1.0, 2.0),
                    "prob": 0.9,
                    "rotate_range": [1, 1, 1],
                    "device": device,
                    "spatial_size": (2, 2, 2),
                    "mode": "bilinear",
                },
                {"img": p(torch.arange(27).reshape((1, 3, 3, 3))), "seg": p(torch.arange(27).reshape((1, 3, 3, 3)))},
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
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "mode": ("bilinear", "nearest"),
                    "magnitude_range": (0.3, 0.3),
                    "sigma_range": (1.0, 2.0),
                    "prob": 0.9,
                    "rotate_range": [1, 1, 1],
                    "device": device,
                    "spatial_size": (2, 2, 2),
                },
                {"img": p(torch.arange(27).reshape((1, 3, 3, 3))), "seg": p(torch.arange(27).reshape((1, 3, 3, 3)))},
                {
                    "img": p(
                        torch.tensor(
                            [[[[5.0069, 9.4639], [9.2878, 13.7397]], [[12.3194, 16.7772], [16.5943, 21.0457]]]]
                        )
                    ),
                    "seg": p(torch.tensor([[[[4.0, 14.0], [7.0, 14.0]], [[9.0, 19.0], [12.0, 22.0]]]])),
                },
            ]
        )


class TestRand3DElasticd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_3d_elasticd(self, input_param, input_data, expected_val):
        g = Rand3DElasticd(**input_param)
        g.set_random_state(123)
        if input_param.get("device", None) is None and isinstance(input_data["img"], torch.Tensor):
            input_data["img"].to("cuda:0" if torch.cuda.is_available() else "cpu")
        res = g(input_data)
        for key in res:
            result = res[key]
            expected = expected_val[key] if isinstance(expected_val, dict) else expected_val
            assert_allclose(result, expected, type_test=False, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
