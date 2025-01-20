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
from torch.autograd import gradcheck

from monai.networks.layers.filtering import BilateralFilter
from tests.util import skip_if_no_cpp_extension

TEST_CASES = [
    [
        # Case Description
        "1 dimension, 1 channel, low spatial sigma, low color sigma",
        # Spatial and Color Sigmas
        (1, 0.2),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [1, 0, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 0]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [1.000000, 0.000000, 0.000000, 0.000000, 1.000000]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000000, 0.000000, 1.000000, 0.000000, 0.000000]
            ],
        ],
    ],
    [
        # Case Description
        "1 dimension, 1 channel, low spatial sigma, high color sigma",
        # Spatial and Color Sigmas
        (1, 0.9),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [1, 0, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 0]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.631360, 0.099349, 0.070177, 0.164534, 0.649869]
            ],
            # Batch 1
            [
                # Channel 0
                [0.052271, 0.173599, 0.481337, 0.183721, 0.045619]
            ],
        ],
    ],
    [
        # Case Description
        "1 dimension, 1 channel, high spatial sigma, low color sigma",
        # Spatial and Color Sigmas
        (4, 0.2),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [1, 0, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 0]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [1.000000, 0.000000, 0.000000, 0.000000, 1.000000]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000000, 0.000000, 1.000000, 0.000000, 0.000000]
            ],
        ],
    ],
    [
        # Case Description
        "1 dimension, 1 channel, high spatial sigma, high color sigma",
        # Sigmas
        (4, 0.9),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [1, 0, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 0]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.497667, 0.268683, 0.265026, 0.261467, 0.495981]
            ],
            # Batch 1
            [
                # Channel 0
                [0.145959, 0.142282, 0.315710, 0.135609, 0.132572]
            ],
        ],
    ],
    [
        # Case Description
        "1 dimension, 4 channel, low spatial sigma, high color sigma",
        # Spatial and Color Sigmas
        (1, 0.9),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [1, 0, 0, 0, 0],
                # Channel 1
                [1, 0, 1, 0, 0],
                # Channel 2
                [0, 0, 1, 0, 1],
                # Channel 3
                [0, 0, 0, 0, 1],
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.960843, 0.073540, 0.027689, 0.002676, 0.000000],
                # Channel 1
                [0.960843, 0.073540, 0.951248, 0.003033, 0.000750],
                # Channel 2
                [0.000000, 0.000000, 0.923559, 0.000357, 0.981324],
                # Channel 3
                [0.000000, 0.000000, 0.000000, 0.000000, 0.980574],
            ]
        ],
    ],
    [
        # Case Description
        "2 dimension, 1 channel, high spatial sigma, high color sigma",
        # Sigmas
        (4, 0.9),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]
            ],
            # Batch 1
            [
                # Channel 0
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    [0.213684, 0.094356, 0.092973, 0.091650, 0.216281],
                    [0.094085, 0.092654, 0.091395, 0.090186, 0.089302],
                    [0.092436, 0.091150, 0.090008, 0.088896, 0.088897],
                    [0.090849, 0.089717, 0.088759, 0.087751, 0.088501],
                    [0.211458, 0.088334, 0.087495, 0.087049, 0.212173],
                ]
            ],
            # Batch 1
            [
                # Channel 0
                [
                    [0.033341, 0.031314, 0.029367, 0.027494, 0.025692],
                    [0.031869, 0.030632, 0.028820, 0.027074, 0.025454],
                    [0.030455, 0.029628, 0.084257, 0.026704, 0.025372],
                    [0.029095, 0.028391, 0.027790, 0.026375, 0.025292],
                    [0.027786, 0.027197, 0.026692, 0.026181, 0.025213],
                ]
            ],
        ],
    ],
    [
        # Case Description
        "2 dimension, 4 channel, high spatial sigma, high color sigma",
        # Spatial and Color Sigmas
        (4, 0.9),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
                # Channel 1
                [[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 1]],
                # Channel 2
                [[0, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]],
                # Channel 3
                [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    [0.244373, 0.014488, 0.036589, 0.014226, 0.024329],
                    [0.014108, 0.014228, 0.014096, 0.013961, 0.013823],
                    [0.013574, 0.013757, 0.013836, 0.013699, 0.013558],
                    [0.013008, 0.013211, 0.013404, 0.013438, 0.013295],
                    [0.025179, 0.012634, 0.034555, 0.013050, 0.237582],
                ],
                # Channel 1
                [
                    [0.271496, 0.015547, 0.439432, 0.015700, 0.089579],
                    [0.015252, 0.015702, 0.015779, 0.015859, 0.015940],
                    [0.015020, 0.015556, 0.015935, 0.016015, 0.016098],
                    [0.014774, 0.015331, 0.015860, 0.016171, 0.016255],
                    [0.107384, 0.015094, 0.462471, 0.016166, 0.263480],
                ],
                # Channel 2
                [
                    [0.027123, 0.003527, 0.467273, 0.004912, 0.645776],
                    [0.003810, 0.004908, 0.005605, 0.006319, 0.007050],
                    [0.004816, 0.005991, 0.006989, 0.007716, 0.008459],
                    [0.005880, 0.007060, 0.008179, 0.009101, 0.009858],
                    [0.633398, 0.008191, 0.496893, 0.010376, 0.025898],
                ],
                # Channel 3
                [
                    [0.000000, 0.002468, 0.064430, 0.003437, 0.580526],
                    [0.002666, 0.003434, 0.003922, 0.004422, 0.004933],
                    [0.003370, 0.004192, 0.004890, 0.005399, 0.005919],
                    [0.004115, 0.004940, 0.005723, 0.006368, 0.006898],
                    [0.551194, 0.005731, 0.068977, 0.007260, 0.000000],
                ],
            ]
        ],
    ],
    [
        # Case Description
        "3 dimension, 1 channel, high spatial sigma, high color sigma",
        # Sigmas
        (4, 0.9),
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Frame 0
                    [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
                    # Frame 1
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    # Frame 2
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    # Frame 3
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    # Frame 4
                    [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
                ]
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Frame 0
                    [
                        [0.086801, 0.036670, 0.035971, 0.035304, 0.088456],
                        [0.036639, 0.035652, 0.035009, 0.034394, 0.033803],
                        [0.035899, 0.034897, 0.034136, 0.033566, 0.033129],
                        [0.035180, 0.034238, 0.033413, 0.032811, 0.032577],
                        [0.088290, 0.033597, 0.032821, 0.032134, 0.088786],
                    ],
                    # Frame 1
                    [
                        [0.036286, 0.035269, 0.034632, 0.034021, 0.033435],
                        [0.035398, 0.034485, 0.033922, 0.033381, 0.033177],
                        [0.034688, 0.033822, 0.033169, 0.032664, 0.032780],
                        [0.034024, 0.033234, 0.032533, 0.032005, 0.032388],
                        [0.033564, 0.032797, 0.032118, 0.031525, 0.032105],
                    ],
                    # Frame 2
                    [
                        [0.035225, 0.034169, 0.033404, 0.032843, 0.032766],
                        [0.034383, 0.033487, 0.032908, 0.032415, 0.032650],
                        [0.033691, 0.032921, 0.032353, 0.031900, 0.032384],
                        [0.033080, 0.032390, 0.031786, 0.031432, 0.032008],
                        [0.033099, 0.032373, 0.031737, 0.031479, 0.032054],
                    ],
                    # Frame 3
                    [
                        [0.034216, 0.033231, 0.032337, 0.031758, 0.032101],
                        [0.033456, 0.032669, 0.031913, 0.031455, 0.032034],
                        [0.032788, 0.032140, 0.031618, 0.031413, 0.031977],
                        [0.032221, 0.031650, 0.031145, 0.031130, 0.031652],
                        [0.032642, 0.031968, 0.031378, 0.031433, 0.032003],
                    ],
                    # Frame 4
                    [
                        [0.086207, 0.032335, 0.031499, 0.030832, 0.087498],
                        [0.032570, 0.031884, 0.031155, 0.030858, 0.031401],
                        [0.031967, 0.031417, 0.030876, 0.030881, 0.031388],
                        [0.031602, 0.031103, 0.030696, 0.030960, 0.031455],
                        [0.090599, 0.031546, 0.031127, 0.031386, 0.083483],
                    ],
                ]
            ]
        ],
    ],
]


@skip_if_no_cpp_extension
class BilateralFilterTestCaseCpuApprox(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cpu_approx(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")
        fast_approx = True

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        output = BilateralFilter.apply(input_tensor, *sigmas, fast_approx).cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cpu_approx_backwards(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")
        fast_approx = True

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        input_tensor.requires_grad = True

        # Prepare args
        args = (input_tensor, *sigmas, fast_approx)

        # Run grad check
        gradcheck(BilateralFilter.apply, args, raise_exception=False)


if __name__ == "__main__":
    unittest.main()
