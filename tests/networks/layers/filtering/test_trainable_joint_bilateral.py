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

from monai.networks.layers.filtering import TrainableJointBilateralFilterFunction
from tests.util import skip_if_no_cpp_extension, skip_if_no_cuda, skip_if_quick

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
        # Guide
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 1]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.622459, 0.377540, 0.000001, 0.000001, 0.999997]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000000, 0.000001, 0.880793, 0.000002, 0.119203]
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
        # Guide
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 1]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.595404, 0.302253, 0.070203, 0.163038, 0.714200]
            ],
            # Batch 1
            [
                # Channel 0
                [0.043465, 0.158126, 0.536864, 0.182809, 0.092537]
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
        # Guide
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 1]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.623709, 0.632901, 0.000003, 0.000003, 0.680336]
            ],
            # Batch 1
            [
                # Channel 0
                [0.000001, 0.000001, 0.531206, 0.000001, 0.468788]
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
        # Guide
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 0, 1]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0.464455, 0.463098, 0.276430, 0.275530, 0.478105]
            ],
            # Batch 1
            [
                # Channel 0
                [0.134956, 0.138247, 0.293759, 0.141954, 0.281082]
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
        # Guide
        [
            # Batch 0
            [
                # Channel 0
                [[1, 1, 0, 0, 1], [0, 0, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1]]
            ],
            # Batch 1
            [
                # Channel 0
                [[0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    [0.186535, 0.187357, 0.105377, 0.103652, 0.198665],
                    [0.112617, 0.108847, 0.105970, 0.189602, 0.102954],
                    [0.178338, 0.179829, 0.107473, 0.105256, 0.103963],
                    [0.117651, 0.113304, 0.109876, 0.107392, 0.105853],
                    [0.121557, 0.177689, 0.113150, 0.110388, 0.192877],
                ]
            ],
            # Batch 1
            [
                # Channel 0
                [
                    [0.047156, 0.047865, 0.048233, 0.038611, 0.047911],
                    [0.047607, 0.048292, 0.048633, 0.039251, 0.038611],
                    [0.047715, 0.048369, 0.048678, 0.048633, 0.048233],
                    [0.047477, 0.048094, 0.048369, 0.048292, 0.047865],
                    [0.039190, 0.047477, 0.047715, 0.047607, 0.047156],
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
        # Guide
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Frame 0
                    [[1, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1]],
                    # Frame 1
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    # Frame 2
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    # Frame 3
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    # Frame 4
                    [[1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 1, 1]],
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
                        [0.089316, 0.088903, 0.033707, 0.033461, 0.091881],
                        [0.035173, 0.034324, 0.033747, 0.033448, 0.033427],
                        [0.035619, 0.034710, 0.034074, 0.033720, 0.033646],
                        [0.036364, 0.035387, 0.034688, 0.034275, 0.034148],
                        [0.037401, 0.085687, 0.035583, 0.035109, 0.089741],
                    ],
                    # Frame 1
                    [
                        [0.034248, 0.033502, 0.033023, 0.032816, 0.032881],
                        [0.034339, 0.033546, 0.033020, 0.032767, 0.032785],
                        [0.034721, 0.033876, 0.033298, 0.032994, 0.032965],
                        [0.035397, 0.034490, 0.033856, 0.033501, 0.033424],
                        [0.036357, 0.035383, 0.034688, 0.034279, 0.034155],
                    ],
                    # Frame 2
                    [
                        [0.033748, 0.033047, 0.032609, 0.032441, 0.032541],
                        [0.033782, 0.033041, 0.032562, 0.032353, 0.032410],
                        [0.034104, 0.033316, 0.032792, 0.032538, 0.032554],
                        [0.034714, 0.033872, 0.033298, 0.032998, 0.032972],
                        [0.035604, 0.034702, 0.034074, 0.033727, 0.033660],
                    ],
                    # Frame 3
                    [
                        [0.033533, 0.032871, 0.032471, 0.032340, 0.032476],
                        [0.033511, 0.032815, 0.032380, 0.032212, 0.032310],
                        [0.033775, 0.033037, 0.032562, 0.032356, 0.032417],
                        [0.034324, 0.033539, 0.033020, 0.032774, 0.032799],
                        [0.035151, 0.034313, 0.033747, 0.033459, 0.033449],
                    ],
                    # Frame 4
                    [
                        [0.091383, 0.090681, 0.032608, 0.091418, 0.092851],
                        [0.033525, 0.032867, 0.032471, 0.032344, 0.032483],
                        [0.033733, 0.033039, 0.032609, 0.032448, 0.032555],
                        [0.034226, 0.033491, 0.033023, 0.032827, 0.032903],
                        [0.089445, 0.034216, 0.033707, 0.090126, 0.091748],
                    ],
                ]
            ]
        ],
    ],
]


@skip_if_no_cpp_extension
@skip_if_quick
class JointBilateralFilterTestCaseCpuPrecise(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cpu_precise(self, test_case_description, sigmas, input, guide, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)
        guide_tensor = torch.from_numpy(np.array(guide)).to(dtype=torch.double, device=device)

        len_input = len(input_tensor.shape)
        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(4)

        output = TrainableJointBilateralFilterFunction.apply(input_tensor, guide_tensor, *sigmas).cpu().numpy()

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            output = output.squeeze(4).squeeze(3)
        elif len_input == 4:
            output = output.squeeze(4)

        # Ensure result are as expected.
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cpu_precise_backwards(self, test_case_description, sigmas, input, guide, expected):
        # Params to determine the implementation to test
        device = torch.device("cpu")

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)
        input_tensor.requires_grad = True
        guide_tensor = torch.from_numpy(np.array(guide)).to(dtype=torch.double, device=device)

        # C++ extension so far only supports 5-dim inputs.
        len_input = len(input_tensor.shape)
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(4)

        # Check gradient toward input.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-6, atol=1e-5, raise_exception=False)
        input_tensor = input_tensor.detach()
        input_tensor.requires_grad = False

        # Check gradient toward guide.
        guide_tensor.requires_grad = True
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-6, atol=1e-5, raise_exception=False)
        guide_tensor = guide_tensor.detach()
        guide_tensor.guide_tensor = False

        # Check gradient toward sigma_x.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_y.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_z.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_color.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3], dtype=torch.double, requires_grad=True),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-3, atol=1e-3, raise_exception=False)


@skip_if_no_cuda
@skip_if_no_cpp_extension
class JointBilateralFilterTestCaseCudaPrecise(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_cuda_precise(self, test_case_description, sigmas, input, guide, expected):
        # Skip this test
        if not torch.cuda.is_available():
            return

        # Params to determine the implementation to test
        device = torch.device("cuda")

        # Create input tensor and apply filter
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)
        guide_tensor = torch.from_numpy(np.array(guide)).to(dtype=torch.double, device=device)

        len_input = len(input_tensor.shape)
        # C++ extension so far only supports 5-dim inputs.
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(4)

        output = TrainableJointBilateralFilterFunction.apply(input_tensor, guide_tensor, *sigmas).cpu().numpy()

        # Make sure to return tensor of the same shape as the input.
        if len_input == 3:
            output = output.squeeze(4).squeeze(3)
        elif len_input == 4:
            output = output.squeeze(4)

        # Ensure result are as expected.
        np.testing.assert_allclose(output, expected, atol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_cuda_precise_backwards(self, test_case_description, sigmas, input, guide, expected):
        # Params to determine the implementation to test
        device = torch.device("cuda")

        # Prepare input tensor
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.double, device=device)
        input_tensor.requires_grad = True
        guide_tensor = torch.from_numpy(np.array(guide)).to(dtype=torch.double, device=device)

        # C++ extension so far only supports 5-dim inputs.
        len_input = len(input_tensor.shape)
        if len_input == 3:
            input_tensor = input_tensor.unsqueeze(3).unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(3).unsqueeze(4)
        elif len_input == 4:
            input_tensor = input_tensor.unsqueeze(4)
            guide_tensor = guide_tensor.unsqueeze(4)

        # Check gradient toward input.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-6, atol=1e-5, raise_exception=False)
        input_tensor = input_tensor.detach()
        input_tensor.requires_grad = False

        # Check gradient toward guide.
        guide_tensor.requires_grad = True
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-6, atol=1e-5, raise_exception=False)
        guide_tensor = guide_tensor.detach()
        guide_tensor.guide_tensor = False

        # Check gradient toward sigma_x.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_y.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_z.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2], dtype=torch.double, requires_grad=True),
            torch.tensor(sigmas[3]),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-2, atol=1e-3, raise_exception=False)

        # Check gradient toward sigma_color.
        args = (
            input_tensor,
            guide_tensor,
            torch.tensor(sigmas[0]),
            torch.tensor(sigmas[1]),
            torch.tensor(sigmas[2]),
            torch.tensor(sigmas[3], dtype=torch.double, requires_grad=True),
        )
        gradcheck(TrainableJointBilateralFilterFunction.apply, args, eps=1e-3, atol=1e-3, raise_exception=False)


if __name__ == "__main__":
    unittest.main()
