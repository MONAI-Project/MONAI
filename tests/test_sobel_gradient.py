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

import unittest

import torch
from parameterized import parameterized

from monai.transforms import SobelGradients
from tests.utils import assert_allclose

IMAGE = torch.zeros(1, 16, 16, dtype=torch.float32)
IMAGE[0, 8, :] = 1
OUTPUT_3x3 = torch.zeros(2, 16, 16, dtype=torch.float32)
OUTPUT_3x3[0, 7, :] = 2.0
OUTPUT_3x3[0, 9, :] = -2.0
OUTPUT_3x3[0, 7, 0] = OUTPUT_3x3[0, 7, -1] = 1.5
OUTPUT_3x3[0, 9, 0] = OUTPUT_3x3[0, 9, -1] = -1.5
OUTPUT_3x3[1, 7, 0] = OUTPUT_3x3[1, 9, 0] = 0.5
OUTPUT_3x3[1, 8, 0] = 1.0
OUTPUT_3x3[1, 8, -1] = -1.0
OUTPUT_3x3[1, 7, -1] = OUTPUT_3x3[1, 9, -1] = -0.5

TEST_CASE_0 = [IMAGE, {"kernel_size": 3, "dtype": torch.float32}, OUTPUT_3x3]
TEST_CASE_1 = [IMAGE, {"kernel_size": 3, "dtype": torch.float64}, OUTPUT_3x3]
TEST_CASE_2 = [IMAGE, {"kernel_size": 3, "direction": "horizontal", "dtype": torch.float64}, OUTPUT_3x3[0][None, ...]]
TEST_CASE_3 = [IMAGE, {"kernel_size": 3, "direction": "vertical", "dtype": torch.float64}, OUTPUT_3x3[1][None, ...]]
TEST_CASE_4 = [IMAGE, {"kernel_size": 3, "direction": ["vertical"], "dtype": torch.float64}, OUTPUT_3x3[1][None, ...]]
TEST_CASE_5 = [IMAGE, {"kernel_size": 3, "direction": ["horizontal", "vertical"], "dtype": torch.float64}, OUTPUT_3x3]
TEST_CASE_6 = [IMAGE, {"kernel_size": 3, "direction": ("horizontal", "vertical"), "dtype": torch.float64}, OUTPUT_3x3]

TEST_CASE_KERNEL_0 = [
    {"kernel_size": 3, "dtype": torch.float64},
    torch.tensor([[-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float64),
]
TEST_CASE_KERNEL_1 = [
    {"kernel_size": 5, "dtype": torch.float64},
    torch.tensor(
        [
            [-0.25, -0.2, 0.0, 0.2, 0.25],
            [-0.4, -0.5, 0.0, 0.5, 0.4],
            [-0.5, -1.0, 0.0, 1.0, 0.5],
            [-0.4, -0.5, 0.0, 0.5, 0.4],
            [-0.25, -0.2, 0.0, 0.2, 0.25],
        ],
        dtype=torch.float64,
    ),
]
TEST_CASE_KERNEL_2 = [
    {"kernel_size": 7, "dtype": torch.float64},
    torch.tensor(
        [
            [-3.0 / 18.0, -2.0 / 13.0, -1.0 / 10.0, 0.0, 1.0 / 10.0, 2.0 / 13.0, 3.0 / 18.0],
            [-3.0 / 13.0, -2.0 / 8.0, -1.0 / 5.0, 0.0, 1.0 / 5.0, 2.0 / 8.0, 3.0 / 13.0],
            [-3.0 / 10.0, -2.0 / 5.0, -1.0 / 2.0, 0.0, 1.0 / 2.0, 2.0 / 5.0, 3.0 / 10.0],
            [-3.0 / 9.0, -2.0 / 4.0, -1.0 / 1.0, 0.0, 1.0 / 1.0, 2.0 / 4.0, 3.0 / 9.0],
            [-3.0 / 10.0, -2.0 / 5.0, -1.0 / 2.0, 0.0, 1.0 / 2.0, 2.0 / 5.0, 3.0 / 10.0],
            [-3.0 / 13.0, -2.0 / 8.0, -1.0 / 5.0, 0.0, 1.0 / 5.0, 2.0 / 8.0, 3.0 / 13.0],
            [-3.0 / 18.0, -2.0 / 13.0, -1.0 / 10.0, 0.0, 1.0 / 10.0, 2.0 / 13.0, 3.0 / 18.0],
        ],
        dtype=torch.float64,
    ),
]
TEST_CASE_ERROR_0 = [{"kernel_size": 1}]  # kernel size less than 3
TEST_CASE_ERROR_1 = [{"kernel_size": 4}]  # even kernel size
TEST_CASE_ERROR_2 = [{"direction": 1}]  # wrong type direction
TEST_CASE_ERROR_3 = [{"direction": "not_exist_direction"}]  # wrong direction
TEST_CASE_ERROR_4 = [{"direction": ["not_exist_direction"]}]  # wrong direction in a list
TEST_CASE_ERROR_5 = [{"direction": ["horizontal", "not_exist_direction"]}]  # correct and wrong direction in a list

TEST_CASE_IMAGE_ERROR_0 = [torch.cat([IMAGE, IMAGE], dim=0), {"kernel_size": 3, "dtype": torch.float32}]


class SobelGradientTests(unittest.TestCase):
    backend = None

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_sobel_gradients(self, image, arguments, expected_grad):
        sobel = SobelGradients(**arguments)
        grad = sobel(image)
        assert_allclose(grad, expected_grad)

    @parameterized.expand([TEST_CASE_KERNEL_0, TEST_CASE_KERNEL_1, TEST_CASE_KERNEL_2])
    def test_sobel_kernels(self, arguments, expected_kernel):
        sobel = SobelGradients(**arguments)
        self.assertTrue(sobel.kernel.dtype == expected_kernel.dtype)
        assert_allclose(sobel.kernel, expected_kernel)

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
    def test_sobel_gradients_error(self, arguments):
        with self.assertRaises(ValueError):
            SobelGradients(**arguments)

    @parameterized.expand([TEST_CASE_IMAGE_ERROR_0])
    def test_sobel_gradients_image_error(self, image, arguments):
        sobel = SobelGradients(**arguments)
        with self.assertRaises(ValueError):
            sobel(image)


if __name__ == "__main__":
    unittest.main()
