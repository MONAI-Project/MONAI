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

from monai.transforms import MapLabelValued
from tests.util import assert_allclose

TEST_CASE_1 = [
    {"keys": "seg", "orig_labels": [3, 2, 1], "target_labels": [0, 1, 2]},
    {"seg": np.array([[3, 1], [1, 2]])},
    np.array([[0, 2], [2, 1]]),
]

TEST_CASE_2 = [
    {"keys": "seg", "orig_labels": [3, 5, 8], "target_labels": [0, 1, 2]},
    {"seg": np.array([[[3], [5], [5], [8]]])},
    np.array([[[0], [1], [1], [2]]]),
]

TEST_CASE_3 = [
    {"keys": "seg", "orig_labels": [1, 2, 3], "target_labels": [0, 1, 2]},
    {"seg": np.array([3, 1, 1, 2])},
    np.array([2, 0, 0, 1]),
]

TEST_CASE_4 = [
    {"keys": "seg", "orig_labels": [1, 2, 3], "target_labels": [0.5, 1.5, 2.5]},
    {"seg": np.array([3, 1, 1, 2])},
    np.array([2.5, 0.5, 0.5, 1.5]),
]

TEST_CASE_5 = [
    {"keys": "seg", "orig_labels": [1.5, 2.5, 3.5], "target_labels": [0, 1, 2], "dtype": np.int8},
    {"seg": np.array([3.5, 1.5, 1.5, 2.5])},
    np.array([2, 0, 0, 1]),
]
TEST_CASE_5_1 = [
    {"keys": "seg", "orig_labels": [1.5, 2.5, 3.5], "target_labels": [0, 1, 2], "dtype": torch.int8},
    {"seg": torch.as_tensor([3.5, 1.5, 1.5, 2.5])},
    torch.as_tensor([2.0, 0.0, 0.0, 1.0]),
]

TEST_CASE_6 = [
    {"keys": "seg", "orig_labels": ["label3", "label2", "label1"], "target_labels": [0, 1, 2]},
    {"seg": np.array([["label3", "label1"], ["label1", "label2"]])},
    np.array([[0, 2], [2, 1]]),
]

TEST_CASE_7 = [
    {"keys": "seg", "orig_labels": [3.5, 2.5, 1.5], "target_labels": ["label0", "label1", "label2"], "dtype": "str"},
    {"seg": np.array([[3.5, 1.5], [1.5, 2.5]])},
    np.array([["label0", "label2"], ["label2", "label1"]]),
]


class TestMapLabelValued(unittest.TestCase):
    @parameterized.expand(
        [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_5_1, TEST_CASE_6, TEST_CASE_7]
    )
    def test_shape(self, input_param, input_data, expected_value):
        result = MapLabelValued(**input_param)(input_data)
        if isinstance(expected_value, torch.Tensor):
            assert_allclose(result["seg"], expected_value)
        else:
            np.testing.assert_equal(result["seg"], expected_value)
        self.assertTupleEqual(result["seg"].shape, expected_value.shape)


if __name__ == "__main__":
    unittest.main()
