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
from tests.util import skip_if_no_cpp_extension, skip_if_no_cuda, skip_if_quick

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
                [0.999998, 0.000002, 0.000000, 0.000002, 0.999998]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000000, 0.000001, 0.999995, 0.000001, 0.000000]
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
                [0.813183, 0.186817, 0.061890, 0.186817, 0.813183]
            ],
            # Batch 1
            [
                # Channel 0
                [0.030148, 0.148418, 0.555452, 0.148418, 0.030148]
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
                [0.999999, 0.000009, 0.000009, 0.000009, 0.999999]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000000, 0.000000, 0.999967, 0.000000, 0.000000]
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
                [0.839145, 0.572834, 0.562460, 0.572834, 0.839145]
            ],
            # Batch 1
            [
                # Channel 0
                [0.049925, 0.055062, 0.171732, 0.055062, 0.049925]
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
                [0.889742, 0.141296, 0.027504, 0.000000, 0.000000],
                # Channel 1
                [0.909856, 0.256817, 0.725970, 0.115520, 0.020114],
                # Channel 2
                [0.020114, 0.115520, 0.725970, 0.256817, 0.909856],
                # Channel 3
                [0.000000, 0.000000, 0.027504, 0.141296, 0.889742],
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
                    [0.688943, 0.374599, 0.368574, 0.374599, 0.688943],
                    [0.374599, 0.358248, 0.352546, 0.358248, 0.374599],
                    [0.368574, 0.352546, 0.346955, 0.352546, 0.368574],
                    [0.374599, 0.358248, 0.352546, 0.358248, 0.374599],
                    [0.688943, 0.374599, 0.368574, 0.374599, 0.688943],
                ]
            ],
            # Batch 1
            [
                # Channel 0
                [
                    [0.004266, 0.004687, 0.004836, 0.004687, 0.004266],
                    [0.004687, 0.005150, 0.005314, 0.005150, 0.004687],
                    [0.004836, 0.005314, 0.018598, 0.005314, 0.004836],
                    [0.004687, 0.005150, 0.005314, 0.005150, 0.004687],
                    [0.004266, 0.004687, 0.004836, 0.004687, 0.004266],
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
                    [0.692549, 0.149979, 0.220063, 0.115840, 0.035799],
                    [0.148403, 0.133935, 0.123253, 0.116828, 0.114623],
                    [0.128773, 0.122804, 0.120731, 0.122804, 0.128773],
                    [0.114623, 0.116828, 0.123253, 0.133935, 0.148403],
                    [0.035799, 0.115840, 0.220063, 0.149979, 0.692549],
                ],
                # Channel 1
                [
                    [0.731597, 0.186319, 0.436069, 0.152181, 0.074847],
                    [0.180049, 0.168217, 0.158453, 0.151110, 0.146269],
                    [0.159760, 0.156381, 0.155211, 0.156381, 0.159760],
                    [0.146269, 0.151110, 0.158453, 0.168217, 0.180049],
                    [0.074847, 0.152181, 0.436068, 0.186319, 0.731597],
                ],
                # Channel 2
                [
                    [0.074847, 0.152181, 0.436068, 0.186319, 0.731597],
                    [0.146269, 0.151110, 0.158453, 0.168217, 0.180049],
                    [0.159760, 0.156381, 0.155211, 0.156381, 0.159760],
                    [0.180049, 0.168217, 0.158453, 0.151110, 0.146269],
                    [0.731597, 0.186319, 0.436069, 0.152181, 0.074847],
                ],
                # Channel 3
                [
                    [0.035799, 0.115840, 0.220063, 0.149979, 0.692549],
                    [0.114623, 0.116828, 0.123253, 0.133935, 0.148403],
                    [0.128773, 0.122804, 0.120731, 0.122804, 0.128773],
                    [0.148403, 0.133935, 0.123253, 0.116828, 0.114623],
                    [0.692549, 0.149979, 0.220063, 0.115840, 0.035799],
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
                        [0.554430, 0.254995, 0.251207, 0.254996, 0.554430],
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                        [0.251207, 0.241082, 0.237534, 0.241082, 0.251207],
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                        [0.554430, 0.254995, 0.251207, 0.254996, 0.554430],
                    ],
                    # Frame 1
                    [
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                        [0.244692, 0.234873, 0.231432, 0.234873, 0.244692],
                        [0.241082, 0.231431, 0.228049, 0.231432, 0.241082],
                        [0.244692, 0.234873, 0.231432, 0.234873, 0.244692],
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                    ],
                    # Frame 2
                    [
                        [0.251207, 0.241081, 0.237534, 0.241082, 0.251207],
                        [0.241082, 0.231431, 0.228049, 0.231432, 0.241082],
                        [0.237534, 0.228048, 0.224724, 0.228049, 0.237534],
                        [0.241082, 0.231431, 0.228049, 0.231432, 0.241082],
                        [0.251207, 0.241081, 0.237534, 0.241082, 0.251207],
                    ],
                    # Frame 3
                    [
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                        [0.244692, 0.234873, 0.231432, 0.234873, 0.244692],
                        [0.241082, 0.231431, 0.228049, 0.231432, 0.241082],
                        [0.244692, 0.234873, 0.231432, 0.234873, 0.244692],
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                    ],
                    # Frame 4
                    [
                        [0.554430, 0.254995, 0.251207, 0.254996, 0.554430],
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                        [0.251207, 0.241082, 0.237534, 0.241082, 0.251207],
                        [0.254996, 0.244691, 0.241082, 0.244692, 0.254996],
                        [0.554430, 0.254995, 0.251207, 0.254996, 0.554430],
                    ],
                ]
            ]
        ],
    ],
]


@skip_if_no_cpp_extension
@skip_if_quick
class BilateralFilterTestCaseCpuPrecise(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cpu_precise(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")
        fast_approx = False

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        output = BilateralFilter.apply(input_tensor, *sigmas, fast_approx).cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cpu_precise_backwards(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")
        fast_approx = False

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        input_tensor.requires_grad = True

        # Prepare args
        args = (input_tensor, *sigmas, fast_approx)

        # Run grad check
        gradcheck(BilateralFilter.apply, args, raise_exception=False)


@skip_if_no_cuda
@skip_if_no_cpp_extension
class BilateralFilterTestCaseCudaPrecise(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cuda_precise(self, test_case_description, sigmas, input, expected):
        # Skip this test
        if not torch.cuda.is_available():
            return

        # Params to determine the implementation to test
        device = torch.device("cuda")
        fast_approx = False

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        output = BilateralFilter.apply(input_tensor, *sigmas, fast_approx).cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cuda_precise_backwards(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cuda")
        fast_approx = False

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=device)
        input_tensor.requires_grad = True

        # Prepare args
        args = (input_tensor, *sigmas, fast_approx)

        # Run grad check
        gradcheck(BilateralFilter.apply, args, raise_exception=False)


if __name__ == "__main__":
    unittest.main()
