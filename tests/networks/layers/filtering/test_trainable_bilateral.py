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

from monai.networks.layers.filtering import TrainableBilateralFilterFunction
from tests.util import skip_if_no_cpp_extension, skip_if_no_cuda

TEST_CASES = [
    [
        # Case Description
        "1 dimension, 1 channel, low spatial sigmas, low color sigma",
        # (sigma_x, sigma_y, sigma_z, color_sigma)
        (1.0, 1.0, 1.0, 0.2),
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
                [0.999997, 0.000001, 0.000000, 0.000001, 0.999997]
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
        "1 dimension, 1 channel, low spatial sigmas, high color sigma",
        # (sigma_x, sigma_y, sigma_z, color_sigma)
        (1, 1, 1, 0.9),
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
                [0.714200, 0.158126, 0.061890, 0.158126, 0.714200]
            ],
            # Batch 1
            [
                # Channel 0
                [0.043465, 0.158126, 0.555452, 0.158126, 0.043465]
            ],
        ],
    ],
    [
        # Case Description
        "1 dimension, 1 channel, high spatial sigmas, low color sigma",
        # (sigma_x, sigma_y, sigma_z, color_sigma)
        (4, 4, 4, 0.2),
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
                [0.999994, 0.000002, 0.000002, 0.000002, 0.999994]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000001, 0.000001, 0.999986, 0.000001, 0.000001]
            ],
        ],
    ],
    [
        # Case Description
        "1 dimension, 1 channel, high spatial sigmas, high color sigma",
        # (sigma_x, sigma_y, sigma_z, color_sigma)
        (4, 4, 4, 0.9),
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
                [0.533282, 0.245915, 0.244711, 0.245915, 0.533282]
            ],
            # Batch 1
            [
                # Channel 0
                [0.125052, 0.126608, 0.333592, 0.126608, 0.125052]
            ],
        ],
    ],
    [
        # Case Description
        "2 dimensions, 1 channel, high spatial sigmas, high color sigma",
        # (sigma_x, sigma_y, sigma_z, color_sigma)
        (4, 4, 4, 0.9),
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
                    [0.239789, 0.082990, 0.082630, 0.082990, 0.239789],
                    [0.082990, 0.081934, 0.081579, 0.081934, 0.082990],
                    [0.082630, 0.081579, 0.081225, 0.081579, 0.082630],
                    [0.082990, 0.081934, 0.081579, 0.081934, 0.082990],
                    [0.239789, 0.082990, 0.082630, 0.082990, 0.239789],
                ]
            ],
            # Batch 1
            [
                # Channel 0
                [
                    [0.024155, 0.024432, 0.024525, 0.024432, 0.024155],
                    [0.024432, 0.024712, 0.024806, 0.024712, 0.024432],
                    [0.024525, 0.024806, 0.080686, 0.024806, 0.024525],
                    [0.024432, 0.024712, 0.024806, 0.024712, 0.024432],
                    [0.024155, 0.024432, 0.024525, 0.024432, 0.024155],
                ]
            ],
        ],
    ],
    [
        # Case Description
        "3 dimensions, 1 channel, high spatial sigmas, high color sigma",
        # (sigma_x, sigma_y, sigma_z, color_sigma)
        (4, 4, 4, 0.9),
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
                        [0.098142, 0.030317, 0.030191, 0.030316, 0.098142],
                        [0.030316, 0.029947, 0.029822, 0.029947, 0.030316],
                        [0.030191, 0.029822, 0.029698, 0.029822, 0.030191],
                        [0.030316, 0.029947, 0.029822, 0.029947, 0.030316],
                        [0.098142, 0.030317, 0.030191, 0.030317, 0.098142],
                    ],
                    # Frame 1
                    [
                        [0.030316, 0.029947, 0.029822, 0.029947, 0.030316],
                        [0.029947, 0.029581, 0.029458, 0.029581, 0.029947],
                        [0.029822, 0.029458, 0.029336, 0.029458, 0.029822],
                        [0.029947, 0.029581, 0.029458, 0.029581, 0.029947],
                        [0.030316, 0.029947, 0.029822, 0.029947, 0.030316],
                    ],
                    # Frame 2
                    [
                        [0.030191, 0.029822, 0.029698, 0.029822, 0.030191],
                        [0.029822, 0.029458, 0.029336, 0.029458, 0.029822],
                        [0.029698, 0.029336, 0.029214, 0.029336, 0.029698],
                        [0.029822, 0.029458, 0.029336, 0.029458, 0.029822],
                        [0.030191, 0.029822, 0.029698, 0.029822, 0.030191],
                    ],
                    # Frame 3
                    [
                        [0.030316, 0.029947, 0.029822, 0.029947, 0.030317],
                        [0.029947, 0.029581, 0.029458, 0.029581, 0.029947],
                        [0.029822, 0.029458, 0.029336, 0.029458, 0.029822],
                        [0.029947, 0.029581, 0.029458, 0.029581, 0.029947],
                        [0.030316, 0.029947, 0.029822, 0.029947, 0.030316],
                    ],
                    # Frame 4
                    [
                        [0.098142, 0.030317, 0.030191, 0.030317, 0.098142],
                        [0.030317, 0.029947, 0.029822, 0.029947, 0.030316],
                        [0.030191, 0.029822, 0.029698, 0.029822, 0.030191],
                        [0.030317, 0.029947, 0.029822, 0.029947, 0.030316],
                        [0.098142, 0.030317, 0.030191, 0.030316, 0.098142],
                    ],
                ]
            ]
        ],
    ],
]


@skip_if_no_cpp_extension
class BilateralFilterTestCaseCpuPrecise(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cpu_precise(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)

        len_input = len(input_tensor.shape)
        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)

        output = TrainableBilateralFilterFunction.apply(input_tensor, *sigmas).cpu().numpy()

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            output = output.squeeze(4).squeeze(3)
        elif len_input == 4:
            output = output.squeeze(4)

        # Ensure result are as expected.
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cpu_precise_backwards(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)
        input_tensor.requires_grad = True

        # C++ extension so far only supports 5-dim inputs.
        len_input = len(input_tensor.shape)
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)

        # Check gradient toward input.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-6, atol=1e-5, raise_exception=False)
        input_tensor = input_tensor.detach()
        input_tensor.requires_grad = False

        # Check gradient toward sigma_x.
        args = (
            input_tensor,
            torch.tensor(sigmas[0], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_y.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_z.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_color.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3], dtype=torch.double, requires_grad=True),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-3, atol=1e-3, raise_exception=False)


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

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)

        len_input = len(input_tensor.shape)
        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)

        output = TrainableBilateralFilterFunction.apply(input_tensor, *sigmas).cpu().numpy()

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            output = output.squeeze(4).squeeze(3)
        elif len_input == 4:
            output = output.squeeze(4)

        # Ensure result are as expected.
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cuda_precise_backwards(self, test_case_description, sigmas, input, expected):
        # Params to determine the implementation to test
        device = torch.device("cuda")

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)
        input_tensor.requires_grad = True

        # C++ extension so far only supports 5-dim inputs.
        len_input = len(input_tensor.shape)
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)

        # Check gradient toward input.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-6, atol=1e-5, raise_exception=False)
        input_tensor = input_tensor.detach()
        input_tensor.requires_grad = False

        # Check gradient toward sigma_x.
        args = (
            input_tensor,
            torch.tensor(sigmas[0], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_y.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_z.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_color.
        args = (
            input_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3], dtype=torch.double, requires_grad=True),
        )
        gradcheck(TrainableBilateralFilterFunction.apply, args, eps=1e-3, atol=1e-3, raise_exception=False)


if __name__ == "__main__":
    unittest.main()
