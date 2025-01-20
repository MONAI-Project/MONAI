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

from monai.transforms import Rand2DElasticd
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, is_tf32_env

_rtol = 5e-3 if is_tf32_env() else 1e-4

TESTS = []
for p in TEST_NDARRAYS_ALL:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (1.0, 2.0),
                    "prob": 0.0,
                    "device": device,
                    "spatial_size": (2, 2),
                },
                {"img": p(torch.ones((3, 3, 3))), "seg": p(torch.ones((3, 3, 3)))},
                p(np.ones((3, 2, 2))),
            ]
        )
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (0.3, 0.3),
                    "prob": 0.0,
                    "device": device,
                    "spatial_size": -1,
                },
                {"img": p(torch.arange(4).reshape((1, 2, 2))), "seg": p(torch.arange(4).reshape((1, 2, 2)))},
                p(np.arange(4).reshape((1, 2, 2))),
            ]
        )
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (1.0, 2.0),
                    "prob": 0.9,
                    "padding_mode": "zeros",
                    "device": device,
                    "spatial_size": (2, 2),
                    "mode": "bilinear",
                },
                {"img": p(torch.ones((3, 3, 3))), "seg": p(torch.ones((3, 3, 3)))},
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
                    "keys": ("img", "seg"),
                    "spacing": (1.0, 1.0),
                    "magnitude_range": (1.0, 1.0),
                    "scale_range": [1.2, 2.2],
                    "prob": 0.9,
                    "padding_mode": "border",
                    "device": device,
                    "spatial_size": (2, 2),
                },
                {"img": p(torch.arange(27).reshape((3, 3, 3))), "seg": p(torch.arange(27).reshape((3, 3, 3)))},
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
                    "keys": ("img", "seg"),
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (0.1, 0.2),
                    "translate_range": [-0.01, 0.01],
                    "scale_range": [0.01, 0.02],
                    "prob": 0.9,
                    "device": device,
                    "spatial_size": (2, 2),
                },
                {"img": p(torch.arange(27).reshape((3, 3, 3))), "seg": p(torch.arange(27).reshape((3, 3, 3)))},
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
        TESTS.append(
            [
                {
                    "keys": ("img", "seg"),
                    "mode": ("bilinear", "nearest"),
                    "spacing": (0.3, 0.3),
                    "magnitude_range": (0.1, 0.2),
                    "translate_range": [-0.01, 0.01],
                    "scale_range": [0.01, 0.02],
                    "prob": 0.9,
                    "device": device,
                    "spatial_size": (2, 2),
                },
                {"img": p(torch.arange(27).reshape((3, 3, 3))), "seg": p(torch.arange(27).reshape((3, 3, 3)))},
                {
                    "img": p(
                        torch.tensor(
                            [
                                [[1.3584, 1.9251], [5.6266, 6.6427]],
                                [[10.3584, 10.9251], [14.6266, 15.6427]],
                                [[19.3584, 19.9251], [23.6266, 24.6427]],
                            ]
                        )
                    ),
                    "seg": p(
                        torch.tensor(
                            [[[0.0, 2.0], [6.0, 8.0]], [[9.0, 11.0], [15.0, 17.0]], [[18.0, 20.0], [24.0, 26.0]]]
                        )
                    ),
                },
            ]
        )


class TestRand2DElasticd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_2d_elasticd(self, input_param, input_data, expected_val):
        g = Rand2DElasticd(**input_param)
        if input_param.get("device", None) is None and isinstance(input_data["img"], torch.Tensor):
            input_data["img"].to("cuda:0" if torch.cuda.is_available() else "cpu")
        g.set_random_state(123)
        res = g(input_data)
        for key in res:
            result = res[key]
            expected = expected_val[key] if isinstance(expected_val, dict) else expected_val
            assert_allclose(result, expected, rtol=_rtol, atol=5e-3, type_test=False)


if __name__ == "__main__":
    unittest.main()
