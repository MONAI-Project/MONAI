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

import random
import unittest

import torch
from parameterized import parameterized

from monai.apps.detection.utils.detector_utils import preprocess_images
from monai.utils import ensure_tuple
from tests.util import assert_allclose

TEST_CASE_1 = [  # 3D, batch 3, 2 input channel
    {
        "pretrained": False,
        "spatial_dims": 3,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": 7,
        "conv1_t_stride": (2, 2, 2),
    },
    (3, 2, 32, 64, 48),
    (3, 2, 64, 64, 64),
]

TEST_CASE_2 = [  # 2D, batch 2, 1 input channel
    {
        "pretrained": False,
        "spatial_dims": 2,
        "n_input_channels": 1,
        "num_classes": 3,
        "conv1_t_size": [7, 7],
        "conv1_t_stride": [2, 2],
    },
    (2, 1, 32, 64),
    (2, 1, 64, 64),
]

TEST_CASE_2_A = [  # 2D, batch 2, 1 input channel, shortcut type A
    {
        "pretrained": False,
        "spatial_dims": 2,
        "n_input_channels": 1,
        "num_classes": 3,
        "shortcut_type": "A",
        "conv1_t_size": (7, 7),
        "conv1_t_stride": 2,
    },
    (2, 1, 32, 64),
    (2, 1, 64, 64),
]

TEST_CASE_3 = [  # 1D, batch 1, 2 input channels
    {
        "pretrained": False,
        "spatial_dims": 1,
        "n_input_channels": 2,
        "num_classes": 3,
        "conv1_t_size": [3],
        "conv1_t_stride": 1,
    },
    (1, 2, 32),
    (1, 2, 32),
]

TEST_CASES = []
TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestDetectorUtils(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_detector_utils(self, input_param, input_shape, expected_shape):
        size_divisible = 32 * ensure_tuple(input_param["conv1_t_stride"])[0]
        input_data = torch.randn(input_shape)
        result, _ = preprocess_images(input_data, input_param["spatial_dims"], size_divisible, mode="constant", value=1)
        assert_allclose(expected_shape, result.shape, type_test=True, device_test=False, atol=0.1)

        input_data = [torch.randn(input_shape[1:]) for _ in range(random.randint(1, 9))]
        result, _ = preprocess_images(input_data, input_param["spatial_dims"], size_divisible, mode="edge")
        expected_shape = (len(input_data),) + expected_shape[1:]
        assert_allclose(expected_shape, result.shape, type_test=True, device_test=False, atol=0.1)


if __name__ == "__main__":
    unittest.main()
