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

from monai.transforms.utils import soft_clip

TEST_CASES = [
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 10},
        {
            "input": torch.arange(10).float(),
            "clipped": torch.tensor([2.0000, 2.0000, 2.0693, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 7.9307, 8.0000]),
        },
    ],
    [
        {"minv": 2, "maxv": None, "sharpness_factor": 10},
        {
            "input": torch.arange(10).float(),
            "clipped": torch.tensor([2.0000, 2.0000, 2.0693, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000]),
        },
    ],
    [
        {"minv": None, "maxv": 7, "sharpness_factor": 10},
        {
            "input": torch.arange(10).float(),
            "clipped": torch.tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 6.9307, 7.0000, 7.0000]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 1.0},
        {
            "input": torch.arange(10).float(),
            "clipped": torch.tensor([2.1266, 2.3124, 2.6907, 3.3065, 4.1088, 5.0000, 5.8912, 6.6935, 7.3093, 7.6877]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 3.0},
        {
            "input": torch.arange(10).float(),
            "clipped": torch.tensor([2.0008, 2.0162, 2.2310, 3.0162, 4.0008, 5.0000, 5.9992, 6.9838, 7.7690, 7.9838]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 5.0},
        {
            "input": torch.arange(10).float(),
            "clipped": torch.tensor([2.0000, 2.0013, 2.1386, 3.0013, 4.0000, 5.0000, 6.0000, 6.9987, 7.8614, 7.9987]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 10},
        {
            "input": np.arange(10).astype(np.float32),
            "clipped": np.array([2.0000, 2.0000, 2.0693, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 7.9307, 8.0000]),
        },
    ],
    [
        {"minv": 2, "maxv": None, "sharpness_factor": 10},
        {
            "input": np.arange(10).astype(float),
            "clipped": np.array([2.0000, 2.0000, 2.0693, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000]),
        },
    ],
    [
        {"minv": None, "maxv": 7, "sharpness_factor": 10},
        {
            "input": np.arange(10).astype(float),
            "clipped": np.array([0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 6.9307, 7.0000, 7.0000]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 1.0},
        {
            "input": np.arange(10).astype(float),
            "clipped": np.array([2.1266, 2.3124, 2.6907, 3.3065, 4.1088, 5.0000, 5.8912, 6.6935, 7.3093, 7.6877]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 3.0},
        {
            "input": np.arange(10).astype(float),
            "clipped": np.array([2.0008, 2.0162, 2.2310, 3.0162, 4.0008, 5.0000, 5.9992, 6.9838, 7.7690, 7.9838]),
        },
    ],
    [
        {"minv": 2, "maxv": 8, "sharpness_factor": 5.0},
        {
            "input": np.arange(10).astype(float),
            "clipped": np.array([2.0000, 2.0013, 2.1386, 3.0013, 4.0000, 5.0000, 6.0000, 6.9987, 7.8614, 7.9987]),
        },
    ],
]


class TestSoftClip(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data):
        outputs = soft_clip(input_data["input"], **input_param)
        expected_val = input_data["clipped"]
        if isinstance(outputs, torch.Tensor):
            np.testing.assert_allclose(
                outputs.detach().cpu().numpy(), expected_val.detach().cpu().numpy(), atol=1e-4, rtol=1e-4
            )
        else:
            np.testing.assert_allclose(outputs, expected_val, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
