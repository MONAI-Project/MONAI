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

import torch
from parameterized import parameterized

from monai.transforms import SobelGradientsd
from tests.util import assert_allclose

IMAGE = torch.zeros(1, 16, 16, dtype=torch.float32)
IMAGE[0, 8, :] = 1

# Output with reflect padding
OUTPUT_3x3 = torch.zeros(2, 16, 16, dtype=torch.float32)
OUTPUT_3x3[0, 7, :] = 0.5
OUTPUT_3x3[0, 9, :] = -0.5

# Output with zero padding
OUTPUT_3x3_ZERO_PAD = OUTPUT_3x3.clone()
OUTPUT_3x3_ZERO_PAD[1, 7, 0] = OUTPUT_3x3_ZERO_PAD[1, 9, 0] = 0.125
OUTPUT_3x3_ZERO_PAD[1, 8, 0] = 0.25
OUTPUT_3x3_ZERO_PAD[1, 7, -1] = OUTPUT_3x3_ZERO_PAD[1, 9, -1] = -0.125
OUTPUT_3x3_ZERO_PAD[1, 8, -1] = -0.25
OUTPUT_3x3_ZERO_PAD[0, 7, 0] = OUTPUT_3x3_ZERO_PAD[0, 7, -1] = 3.0 / 8.0
OUTPUT_3x3_ZERO_PAD[0, 9, 0] = OUTPUT_3x3_ZERO_PAD[0, 9, -1] = -3.0 / 8.0

TEST_CASE_0 = [{"image": IMAGE}, {"keys": "image", "kernel_size": 3, "dtype": torch.float32}, {"image": OUTPUT_3x3}]
TEST_CASE_1 = [{"image": IMAGE}, {"keys": "image", "kernel_size": 3, "dtype": torch.float64}, {"image": OUTPUT_3x3}]
TEST_CASE_2 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "dtype": torch.float32, "new_key_prefix": "sobel_"},
    {"sobel_image": OUTPUT_3x3},
]
TEST_CASE_3 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": 0, "dtype": torch.float32},
    {"image": OUTPUT_3x3[0][None, ...]},
]
TEST_CASE_4 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": 1, "dtype": torch.float32},
    {"image": OUTPUT_3x3[1][None, ...]},
]
TEST_CASE_5 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": [1], "dtype": torch.float32},
    {"image": OUTPUT_3x3[1][None, ...]},
]
TEST_CASE_6 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": [0, 1], "normalize_kernels": True, "dtype": torch.float32},
    {"image": OUTPUT_3x3},
]
TEST_CASE_7 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": (0, 1), "padding_mode": "reflect", "dtype": torch.float32},
    {"image": OUTPUT_3x3},
]
TEST_CASE_8 = [
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": (0, 1), "padding_mode": "zeros", "dtype": torch.float32},
    {"image": OUTPUT_3x3_ZERO_PAD},
]
TEST_CASE_9 = [  # Non-normalized kernels
    {"image": IMAGE},
    {"keys": "image", "kernel_size": 3, "spatial_axes": (0, 1), "normalize_kernels": False, "dtype": torch.float32},
    {"image": OUTPUT_3x3 * 8.0},
]
TEST_CASE_10 = [  # Normalized gradients and normalized kernels
    {"image": IMAGE},
    {
        "keys": "image",
        "kernel_size": 3,
        "spatial_axes": (0, 1),
        "normalize_kernels": True,
        "normalize_gradients": True,
        "dtype": torch.float32,
    },
    {"image": torch.cat([OUTPUT_3x3[0:1] + 0.5, OUTPUT_3x3[1:2]])},
]
TEST_CASE_11 = [  # Normalized gradients but non-normalized kernels
    {"image": IMAGE},
    {
        "keys": "image",
        "kernel_size": 3,
        "spatial_axes": (0, 1),
        "normalize_kernels": False,
        "normalize_gradients": True,
        "dtype": torch.float32,
    },
    {"image": torch.cat([OUTPUT_3x3[0:1] + 0.5, OUTPUT_3x3[1:2]])},
]

TEST_CASE_KERNEL_0 = [
    {"keys": "image", "kernel_size": 3, "dtype": torch.float64},
    (torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64), torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)),
]
TEST_CASE_KERNEL_1 = [
    {"keys": "image", "kernel_size": 5, "dtype": torch.float64},
    (
        torch.tensor([-0.1250, -0.2500, 0.0000, 0.2500, 0.1250], dtype=torch.float64),
        torch.tensor([0.0625, 0.2500, 0.3750, 0.2500, 0.0625], dtype=torch.float64),
    ),
]
TEST_CASE_KERNEL_2 = [
    {"keys": "image", "kernel_size": 7, "dtype": torch.float64},
    (
        torch.tensor([-0.03125, -0.125, -0.15625, 0.0, 0.15625, 0.125, 0.03125], dtype=torch.float64),
        torch.tensor([0.015625, 0.09375, 0.234375, 0.3125, 0.234375, 0.09375, 0.015625], dtype=torch.float64),
    ),
]
TEST_CASE_KERNEL_NON_NORMALIZED_0 = [
    {"keys": "image", "kernel_size": 3, "normalize_kernels": False, "dtype": torch.float64},
    (torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64), torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)),
]
TEST_CASE_KERNEL_NON_NORMALIZED_1 = [
    {"keys": "image", "kernel_size": 5, "normalize_kernels": False, "dtype": torch.float64},
    (
        torch.tensor([-1.0, -2.0, 0.0, 2.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float64),
    ),
]
TEST_CASE_KERNEL_NON_NORMALIZED_2 = [
    {"keys": "image", "kernel_size": 7, "normalize_kernels": False, "dtype": torch.float64},
    (
        torch.tensor([-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=torch.float64),
    ),
]
TEST_CASE_ERROR_0 = [{"image": IMAGE}, {"keys": "image", "kernel_size": 1}]  # kernel size less than 3
TEST_CASE_ERROR_1 = [{"image": IMAGE}, {"keys": "image", "kernel_size": 4}]  # even kernel size
TEST_CASE_ERROR_2 = [{"image": IMAGE}, {"keys": "image", "spatial_axes": "horizontal"}]  # wrong type direction
TEST_CASE_ERROR_3 = [{"image": IMAGE}, {"keys": "image", "spatial_axes": 3}]  # wrong direction
TEST_CASE_ERROR_4 = [{"image": IMAGE}, {"keys": "image", "spatial_axes": [3]}]  # wrong direction in a list
TEST_CASE_ERROR_5 = [
    {"image": IMAGE},
    {"keys": "image", "spatial_axes": [0, 4]},
]  # correct and wrong direction in a list


class SobelGradientTests(unittest.TestCase):
    backend = None

    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
            TEST_CASE_10,
            TEST_CASE_11,
        ]
    )
    def test_sobel_gradients(self, image_dict, arguments, expected_grad):
        sobel = SobelGradientsd(**arguments)
        grad = sobel(image_dict)
        key = "image" if "new_key_prefix" not in arguments else arguments["new_key_prefix"] + arguments["keys"]
        assert_allclose(grad[key], expected_grad[key])

    @parameterized.expand(
        [
            TEST_CASE_KERNEL_0,
            TEST_CASE_KERNEL_1,
            TEST_CASE_KERNEL_2,
            TEST_CASE_KERNEL_NON_NORMALIZED_0,
            TEST_CASE_KERNEL_NON_NORMALIZED_1,
            TEST_CASE_KERNEL_NON_NORMALIZED_2,
        ]
    )
    def test_sobel_kernels(self, arguments, expected_kernels):
        sobel = SobelGradientsd(**arguments)
        self.assertEqual(sobel.kernel_diff.dtype, expected_kernels[0].dtype)
        self.assertEqual(sobel.kernel_smooth.dtype, expected_kernels[0].dtype)
        assert_allclose(sobel.kernel_diff, expected_kernels[0])
        assert_allclose(sobel.kernel_smooth, expected_kernels[1])

    @parameterized.expand(
        [
            TEST_CASE_ERROR_0,
            TEST_CASE_ERROR_1,
            TEST_CASE_ERROR_2,
            TEST_CASE_ERROR_3,
            TEST_CASE_ERROR_4,
            TEST_CASE_ERROR_5,
        ]
    )
    def test_sobel_gradients_error(self, image_dict, arguments):
        with self.assertRaises(ValueError):
            sobel = SobelGradientsd(**arguments)
            sobel(image_dict)


if __name__ == "__main__":
    unittest.main()
