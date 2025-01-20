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

from monai.transforms import FlattenSubKeysd

A = torch.randn(2, 2)
B = torch.randn(3, 3)
C = torch.randn(1, 3)
I = torch.randn(2, 3)
D1 = {"a": A, "b": B}
D2 = {"a": A, "b": B, "c": C}

TEST_CASE_0 = [{"keys": "pred"}, {"image": I, "pred": D1}, {"a": A, "b": B, "image": I}]
TEST_CASE_1 = [{"keys": "pred"}, {"image": I, "pred": D2}, {"a": A, "b": B, "c": C, "image": I}]
TEST_CASE_2 = [{"keys": "pred", "sub_keys": ["a", "b"]}, {"image": I, "pred": D1}, {"a": A, "b": B, "image": I}]
TEST_CASE_3 = [{"keys": "pred", "sub_keys": ["a", "b"]}, {"image": I, "pred": D2}, {"a": A, "b": B, "image": I}]
TEST_CASE_4 = [
    {"keys": "pred", "sub_keys": ["a", "b"], "delete_keys": False},
    {"image": I, "pred": D1},
    {"a": A, "b": B, "image": I, "pred": D1},
]
TEST_CASE_5 = [
    {"keys": "pred", "sub_keys": ["a", "b"], "prefix": "new"},
    {"image": I, "pred": D2},
    {"new_a": A, "new_b": B, "image": I},
]
TEST_CASE_ERROR_1 = [  # error for duplicate key
    {"keys": "pred", "sub_keys": ["a", "b"]},
    {"image": I, "pred": D2, "a": None},
]


class TestFlattenSubKeysd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_dict(self, params, input_data, expected):
        result = FlattenSubKeysd(**params)(input_data)
        self.assertSetEqual(set(result.keys()), set(expected.keys()))
        for k in expected:
            self.assertEqual(id(result[k]), id(expected[k]))

    @parameterized.expand([TEST_CASE_ERROR_1])
    def test_error(self, params, input_data):
        with self.assertRaises(ValueError):
            FlattenSubKeysd(**params)(input_data)


if __name__ == "__main__":
    unittest.main()
