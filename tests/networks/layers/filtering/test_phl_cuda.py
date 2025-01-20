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

from monai.networks.layers.filtering import PHLFilter
from tests.util import skip_if_no_cpp_extension, skip_if_no_cuda

TEST_CASES = [
    [
        # Case Description
        "2 batches, 1 dimensions, 1 channels, 1 features",
        # Sigmas
        [1, 0.2],
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
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [1, 0.2, 0.5, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0.5, 0, 1, 1, 1]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.468968, 0.364596, 0.408200, 0.332579, 0.468968]
            ],
            # Batch 1
            [
                # Channel 0
                [0.202473, 0.176527, 0.220995, 0.220995, 0.220995]
            ],
        ],
    ],
    [
        # Case Description
        "1 batches, 1 dimensions, 3 channels, 1 features",
        # Sigmas
        [1],
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [1, 0, 0, 0, 0],
                # Channel 1
                [0, 0, 0, 0, 1],
                # Channel 2
                [0, 0, 1, 0, 0],
            ]
        ],
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [1, 0.2, 0.5, 0.2, 1]
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.229572, 0.182884, 0.202637, 0.182884, 0.229572],
                # Channel 1
                [0.229572, 0.182884, 0.202637, 0.182884, 0.229572],
                # Channel 2
                [0.201235, 0.208194, 0.205409, 0.208194, 0.201235],
            ]
        ],
    ],
    [
        # Case Description
        "1 batches, 2 dimensions, 1 channels, 3 features",
        # Sigmas
        [5, 3, 3],
        # Input
        [
            # Batch 0
            [
                # Channel 0
                [[9, 9, 0, 0, 0], [9, 9, 0, 0, 0], [9, 9, 0, 0, 0], [9, 9, 6, 6, 6], [9, 9, 6, 6, 6]]
            ]
        ],
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [[9, 9, 0, 0, 0], [9, 9, 0, 0, 0], [9, 9, 0, 0, 0], [9, 9, 6, 6, 6], [9, 9, 6, 6, 6]],
                # Channel 1
                [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                # Channel 2
                [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    [7.792655, 7.511395, 0.953769, 0.860538, 0.912978],
                    [7.758870, 7.426762, 1.164386, 1.050956, 1.121830],
                    [7.733974, 7.429964, 1.405752, 1.244949, 1.320862],
                    [7.712976, 7.429060, 5.789552, 5.594258, 5.371737],
                    [7.701185, 7.492719, 5.860026, 5.538241, 5.281656],
                ]
            ]
        ],
    ],
]


@skip_if_no_cuda
@skip_if_no_cpp_extension
class PHLFilterTestCaseCuda(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cuda(self, test_case_description, sigmas, input, features, expected):
        # Create input tensors
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=torch.device("cuda"))
        feature_tensor = torch.from_numpy(np.array(features)).to(dtype=torch.float, device=torch.device("cuda"))

        # apply filter
        output = PHLFilter.apply(input_tensor, feature_tensor, sigmas).cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(output, expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
