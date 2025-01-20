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
from tests.util import skip_if_no_cpp_extension, skip_if_no_cuda

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
                [0.880626, 0.306148, 0.158734, 0.164534, 0.754386]
            ],
            # Batch 1
            [
                # Channel 0
                [0.019010, 0.104507, 0.605634, 0.183721, 0.045619]
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
                [0.149889, 0.148226, 0.367978, 0.144023, 0.141317]
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
                [0.988107, 0.061340, 0.001565, 0.000011, 0.000000],
                # Channel 1
                [0.988107, 0.061340, 0.998000, 0.000016, 0.000123],
                # Channel 2
                [0.000000, 0.000000, 0.996435, 0.000006, 0.999236],
                # Channel 3
                [0.000000, 0.000000, 0.000000, 0.000000, 0.999113],
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
                    [0.211469, 0.094356, 0.092973, 0.091650, 0.211894],
                    [0.093755, 0.091753, 0.090524, 0.089343, 0.088384],
                    [0.091803, 0.089783, 0.088409, 0.087346, 0.086927],
                    [0.089938, 0.088126, 0.086613, 0.085601, 0.085535],
                    [0.208359, 0.086535, 0.085179, 0.084210, 0.205858],
                ]
            ],
            # Batch 1
            [
                # Channel 0
                [
                    [0.032760, 0.030146, 0.027442, 0.024643, 0.021744],
                    [0.030955, 0.029416, 0.026574, 0.023629, 0.020841],
                    [0.028915, 0.027834, 0.115442, 0.022515, 0.020442],
                    [0.026589, 0.025447, 0.024319, 0.021286, 0.019964],
                    [0.023913, 0.022704, 0.021510, 0.020388, 0.019379],
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
                    [0.557349, 0.011031, 0.001800, 0.011265, 0.000631],
                    [0.009824, 0.010361, 0.010429, 0.010506, 0.010595],
                    [0.008709, 0.009252, 0.009688, 0.009714, 0.009744],
                    [0.007589, 0.008042, 0.008576, 0.008887, 0.008852],
                    [0.000420, 0.006827, 0.001048, 0.007763, 0.190722],
                ],
                # Channel 1
                [
                    [0.614072, 0.011045, 0.925766, 0.011287, 0.007548],
                    [0.009838, 0.010382, 0.010454, 0.010536, 0.010630],
                    [0.008727, 0.009277, 0.009720, 0.009751, 0.009787],
                    [0.007611, 0.008071, 0.008613, 0.008932, 0.008904],
                    [0.027088, 0.006859, 0.950749, 0.007815, 0.230270],
                ],
                # Channel 2
                [
                    [0.056723, 0.000150, 0.973790, 0.000233, 0.990814],
                    [0.000151, 0.000214, 0.000257, 0.000307, 0.000364],
                    [0.000186, 0.000257, 0.000328, 0.000384, 0.000449],
                    [0.000221, 0.000295, 0.000382, 0.000465, 0.000538],
                    [0.993884, 0.000333, 0.984743, 0.000532, 0.039548],
                ],
                # Channel 3
                [
                    [0.000000, 0.000136, 0.049824, 0.000210, 0.983897],
                    [0.000136, 0.000193, 0.000232, 0.000277, 0.000329],
                    [0.000168, 0.000232, 0.000297, 0.000347, 0.000405],
                    [0.000200, 0.000266, 0.000345, 0.000420, 0.000485],
                    [0.967217, 0.000301, 0.035041, 0.000481, 0.000000],
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
                        [0.085451, 0.037820, 0.036880, 0.035978, 0.084296],
                        [0.037939, 0.036953, 0.036155, 0.035385, 0.034640],
                        [0.037167, 0.036302, 0.035603, 0.034931, 0.034465],
                        [0.036469, 0.035724, 0.035137, 0.034572, 0.034480],
                        [0.088942, 0.035193, 0.034682, 0.034266, 0.090568],
                    ],
                    # Frame 1
                    [
                        [0.037125, 0.035944, 0.035103, 0.033429, 0.033498],
                        [0.033380, 0.032653, 0.033748, 0.033073, 0.032549],
                        [0.034834, 0.034001, 0.033500, 0.032902, 0.032560],
                        [0.033972, 0.033554, 0.033220, 0.032765, 0.032570],
                        [0.033590, 0.033222, 0.032927, 0.032689, 0.032629],
                    ],
                    # Frame 2
                    [
                        [0.035635, 0.034468, 0.033551, 0.032818, 0.032302],
                        [0.034523, 0.032830, 0.032146, 0.031536, 0.031149],
                        [0.033612, 0.032011, 0.031664, 0.031128, 0.030839],
                        [0.032801, 0.031668, 0.031529, 0.031198, 0.030978],
                        [0.032337, 0.031550, 0.031419, 0.031383, 0.031211],
                    ],
                    # Frame 3
                    [
                        [0.034300, 0.033236, 0.032239, 0.031517, 0.031133],
                        [0.033357, 0.031842, 0.031035, 0.030471, 0.030126],
                        [0.032563, 0.031094, 0.030156, 0.029703, 0.029324],
                        [0.031850, 0.030505, 0.030027, 0.029802, 0.029461],
                        [0.031555, 0.030121, 0.029943, 0.030000, 0.029700],
                    ],
                    # Frame 4
                    [
                        [0.083156, 0.032122, 0.031204, 0.030380, 0.080582],
                        [0.032296, 0.030936, 0.030170, 0.029557, 0.029124],
                        [0.031617, 0.030293, 0.029377, 0.028886, 0.028431],
                        [0.031084, 0.029859, 0.028839, 0.028439, 0.027973],
                        [0.164616, 0.029457, 0.028484, 0.028532, 0.211082],
                    ],
                ]
            ]
        ],
    ],
]


@skip_if_no_cuda
@skip_if_no_cpp_extension
class BilateralFilterTestCaseCudaApprox(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cuda_approx(self, test_case_description, sigmas, input, expected):
        # Skip this test
        if not torch.cuda.is_available():
            return

        # Params to determine the implementation to test
        device = torch.device("cuda")
        fast_approx = True

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        output = BilateralFilter.apply(input_tensor, *sigmas, fast_approx).cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(output, expected, atol=2e-1)

    @parameterized.expand(TEST_CASES)
    def test_cpu_approx_backwards(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cuda")
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
