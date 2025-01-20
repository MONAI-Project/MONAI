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

from monai.metrics import compute_fp_tp_probs, compute_fp_tp_probs_nd, compute_froc_curve_data, compute_froc_score

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_CASE_1 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8], device=_device),
        "y_coord": torch.tensor([0, 2, 3], device=_device),
        "x_coord": torch.tensor([3, 0, 1], device=_device),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "labels_to_exclude": [2],
        "resolution_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    2,
]

TEST_CASE_2 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "y_coord": torch.tensor([0, 2, 3]),
        "x_coord": torch.tensor([3, 0, 1]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "resolution_level": 0,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_3 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "y_coord": torch.tensor([0, 4, 6]),
        "x_coord": torch.tensor([6, 0, 2]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
        "resolution_level": 1,
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_4 = [
    {
        "fp_probs": np.array([0.8, 0.6]),
        "tp_probs": np.array([1, 1, 0, 0, 0.8, 0.8, 0]),
        "num_targets": 4,
        "num_images": 2,
    },
    (0.25, 0.5, 1, 2, 4, 8),
    0.95833333,
]

TEST_CASE_5 = [
    {
        "fp_probs": torch.tensor([0.8, 0.6]),
        "tp_probs": torch.tensor([1, 1, 0, 0, 0.8, 0.8, 0]),
        "num_targets": 4,
        "num_images": 2,
    },
    (0.25),
    0.75,
]

TEST_CASE_ND_1 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "coords": torch.tensor([[0, 3], [2, 0], [3, 1]]),
        "evaluation_mask": np.array([[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]]),
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]

TEST_CASE_ND_2 = [
    {
        "probs": torch.tensor([1, 0.6, 0.8]),
        "coords": torch.tensor([[0, 0, 3], [1, 2, 0], [0, 3, 1]]),
        "evaluation_mask": np.array(
            [
                [[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]],
                [[0, 0, 1, 1], [2, 2, 0, 0], [0, 3, 3, 0], [0, 3, 3, 3]],
            ]
        ),
    },
    np.array([0.6]),
    np.array([1, 0, 0.8]),
    3,
]


class TestComputeFpTp(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, input_data, expected_fp, expected_tp, expected_num):
        fp_probs, tp_probs, num_tumors = compute_fp_tp_probs(**input_data)
        np.testing.assert_allclose(fp_probs, expected_fp, rtol=1e-5)
        np.testing.assert_allclose(tp_probs, expected_tp, rtol=1e-5)
        np.testing.assert_equal(num_tumors, expected_num)


class TestComputeFpTpNd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_ND_1, TEST_CASE_ND_2])
    def test_value(self, input_data, expected_fp, expected_tp, expected_num):
        fp_probs, tp_probs, num_tumors = compute_fp_tp_probs_nd(**input_data)
        np.testing.assert_allclose(fp_probs, expected_fp, rtol=1e-5)
        np.testing.assert_allclose(tp_probs, expected_tp, rtol=1e-5)
        np.testing.assert_equal(num_tumors, expected_num)


class TestComputeFrocScore(unittest.TestCase):

    @parameterized.expand([TEST_CASE_4, TEST_CASE_5])
    def test_value(self, input_data, thresholds, expected_score):
        fps_per_image, total_sensitivity = compute_froc_curve_data(**input_data)
        score = compute_froc_score(fps_per_image, total_sensitivity, thresholds)
        np.testing.assert_allclose(score, expected_score, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
