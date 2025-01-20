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

from monai.losses import NACLLoss

inputs = torch.tensor(
    [
        [
            [
                [0.1498, 0.1158, 0.3996, 0.3730],
                [0.2155, 0.1585, 0.8541, 0.8579],
                [0.6640, 0.2424, 0.0774, 0.0324],
                [0.0580, 0.2180, 0.3447, 0.8722],
            ],
            [
                [0.3908, 0.9366, 0.1779, 0.1003],
                [0.9630, 0.6118, 0.4405, 0.7916],
                [0.5782, 0.9515, 0.4088, 0.3946],
                [0.7860, 0.3910, 0.0324, 0.9568],
            ],
            [
                [0.0759, 0.0238, 0.5570, 0.1691],
                [0.2703, 0.7722, 0.1611, 0.6431],
                [0.8051, 0.6596, 0.4121, 0.1125],
                [0.5283, 0.6746, 0.5528, 0.7913],
            ],
        ]
    ]
)
targets = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]]])

TEST_CASES = [
    [{"classes": 3, "dim": 2}, {"inputs": inputs, "targets": targets}, 1.1442],
    [{"classes": 3, "dim": 2}, {"inputs": inputs.repeat(4, 1, 1, 1), "targets": targets.repeat(4, 1, 1)}, 1.1442],
    [{"classes": 3, "dim": 2, "kernel_ops": "gaussian"}, {"inputs": inputs, "targets": targets}, 1.1433],
    [{"classes": 3, "dim": 2, "kernel_ops": "gaussian", "sigma": 0.5}, {"inputs": inputs, "targets": targets}, 1.1469],
    [{"classes": 3, "dim": 2, "distance_type": "l2"}, {"inputs": inputs, "targets": targets}, 1.1269],
    [{"classes": 3, "dim": 2, "alpha": 0.2}, {"inputs": inputs, "targets": targets}, 1.1790],
    [
        {"classes": 3, "dim": 3, "kernel_ops": "gaussian"},
        {
            "inputs": torch.tensor(
                [
                    [
                        [
                            [
                                [0.5977, 0.2767, 0.0591, 0.1675],
                                [0.4835, 0.3778, 0.8406, 0.3065],
                                [0.6047, 0.2860, 0.9742, 0.2013],
                                [0.9128, 0.8368, 0.6711, 0.4384],
                            ],
                            [
                                [0.9797, 0.1863, 0.5584, 0.6652],
                                [0.2272, 0.2004, 0.7914, 0.4224],
                                [0.5097, 0.8818, 0.2581, 0.3495],
                                [0.1054, 0.5483, 0.3732, 0.3587],
                            ],
                            [
                                [0.3060, 0.7066, 0.7922, 0.4689],
                                [0.1733, 0.8902, 0.6704, 0.2037],
                                [0.8656, 0.5561, 0.2701, 0.0092],
                                [0.1866, 0.7714, 0.6424, 0.9791],
                            ],
                            [
                                [0.5067, 0.3829, 0.6156, 0.8985],
                                [0.5192, 0.8347, 0.2098, 0.2260],
                                [0.8887, 0.3944, 0.6400, 0.5345],
                                [0.1207, 0.3763, 0.5282, 0.7741],
                            ],
                        ],
                        [
                            [
                                [0.8499, 0.4759, 0.1964, 0.5701],
                                [0.3190, 0.1238, 0.2368, 0.9517],
                                [0.0797, 0.6185, 0.0135, 0.8672],
                                [0.4116, 0.1683, 0.1355, 0.0545],
                            ],
                            [
                                [0.7533, 0.2658, 0.5955, 0.4498],
                                [0.9500, 0.2317, 0.2825, 0.9763],
                                [0.1493, 0.1558, 0.3743, 0.8723],
                                [0.1723, 0.7980, 0.8816, 0.0133],
                            ],
                            [
                                [0.8426, 0.2666, 0.2077, 0.3161],
                                [0.1725, 0.8414, 0.1515, 0.2825],
                                [0.4882, 0.5159, 0.4120, 0.1585],
                                [0.2551, 0.9073, 0.7691, 0.9898],
                            ],
                            [
                                [0.4633, 0.8717, 0.8537, 0.2899],
                                [0.3693, 0.7953, 0.1183, 0.4596],
                                [0.0087, 0.7925, 0.0989, 0.8385],
                                [0.8261, 0.6920, 0.7069, 0.4464],
                            ],
                        ],
                        [
                            [
                                [0.0110, 0.1608, 0.4814, 0.6317],
                                [0.0194, 0.9669, 0.3259, 0.0028],
                                [0.5674, 0.8286, 0.0306, 0.5309],
                                [0.3973, 0.8183, 0.0238, 0.1934],
                            ],
                            [
                                [0.8947, 0.6629, 0.9439, 0.8905],
                                [0.0072, 0.1697, 0.4634, 0.0201],
                                [0.7184, 0.2424, 0.0820, 0.7504],
                                [0.3937, 0.1424, 0.4463, 0.5779],
                            ],
                            [
                                [0.4123, 0.6227, 0.0523, 0.8826],
                                [0.0051, 0.0353, 0.3662, 0.7697],
                                [0.4867, 0.8986, 0.2510, 0.5316],
                                [0.1856, 0.2634, 0.9140, 0.9725],
                            ],
                            [
                                [0.2041, 0.4248, 0.2371, 0.7256],
                                [0.2168, 0.5380, 0.4538, 0.7007],
                                [0.9013, 0.2623, 0.0739, 0.2998],
                                [0.1366, 0.5590, 0.2952, 0.4592],
                            ],
                        ],
                    ]
                ]
            ),
            "targets": torch.tensor(
                [
                    [
                        [[0, 1, 0, 1], [1, 2, 1, 0], [2, 1, 1, 1], [1, 1, 0, 1]],
                        [[2, 1, 0, 2], [1, 2, 0, 2], [1, 0, 1, 1], [1, 1, 0, 0]],
                        [[1, 0, 2, 1], [0, 2, 2, 1], [1, 0, 1, 1], [0, 0, 2, 1]],
                        [[2, 1, 1, 0], [1, 0, 0, 2], [1, 0, 2, 1], [2, 1, 0, 1]],
                    ]
                ]
            ),
        },
        1.15035,
    ],
]


class TestNACLLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        loss = NACLLoss(**input_param)
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
