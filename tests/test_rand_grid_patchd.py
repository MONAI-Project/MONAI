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

import numpy as np
from parameterized import parameterized

from monai.transforms.spatial.dictionary import RandGridPatchd
from monai.utils import set_determinism
from tests.utils import TEST_NDARRAYS, assert_allclose

set_determinism(1234)

A = np.arange(16).repeat(3).reshape(4, 4, 3).transpose(2, 0, 1)
A11 = A[:, :2, :2]
A12 = A[:, :2, 2:]
A21 = A[:, 2:, :2]
A22 = A[:, 2:, 2:]

TEST_CASE_0 = [{"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 0}, {"image": A}, [A11, A12, A21, A22]]
TEST_CASE_1 = [{"patch_size": (2, 2), "min_start_pos": 0, "fix_num_patches": 3}, {"image": A}, [A11, A12, A21]]
TEST_CASE_2 = [
    {"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 0, "fix_num_patches": 5},
    {"image": A},
    [A11, A12, A21, A22, np.zeros((3, 2, 2))],
]
TEST_CASE_3 = [{"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 0}, {"image": A}, [A11, A12, A21, A22]]
TEST_CASE_4 = [{"patch_size": (2, 2)}, {"image": A}, [A11, A12, A21, A22]]
TEST_CASE_5 = [{"patch_size": (2, 2), "min_start_pos": 2, "max_start_pos": 2}, {"image": A}, [A22]]
TEST_CASE_6 = [{"patch_size": (2, 2), "min_start_pos": (0, 2), "max_start_pos": (0, 2)}, {"image": A}, [A12, A22]]
TEST_CASE_7 = [{"patch_size": (2, 2), "min_start_pos": 1, "max_start_pos": 2}, {"image": A}, [A22]]
TEST_CASE_8 = [
    {"patch_size": (2, 2), "min_start_pos": 0, "max_start_pos": 1, "fix_num_patches": 1, "sort_fn": "max"},
    {"image": A},
    [A[:, 1:3, 1:3]],
]
TEST_CASE_9 = [
    {
        "patch_size": (3, 3),
        "min_start_pos": -3,
        "max_start_pos": -1,
        "sort_fn": "min",
        "fix_num_patches": 1,
        "pad_opts": {"constant_values": 255},
    },
    {"image": A},
    [np.pad(A[:, :2, 1:], ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=255)],
]

TEST_SINGLE = []
for p in TEST_NDARRAYS:
    TEST_SINGLE.append([p, *TEST_CASE_0])
    TEST_SINGLE.append([p, *TEST_CASE_1])
    TEST_SINGLE.append([p, *TEST_CASE_2])
    TEST_SINGLE.append([p, *TEST_CASE_3])
    TEST_SINGLE.append([p, *TEST_CASE_4])
    TEST_SINGLE.append([p, *TEST_CASE_5])
    TEST_SINGLE.append([p, *TEST_CASE_6])
    TEST_SINGLE.append([p, *TEST_CASE_7])
    TEST_SINGLE.append([p, *TEST_CASE_8])
    TEST_SINGLE.append([p, *TEST_CASE_9])


class TestSlidingPatch(unittest.TestCase):
    @parameterized.expand(TEST_SINGLE)
    def test_split_patch_single_call(self, in_type, input_parameters, image_dict, expected):
        image_key = "image"
        input_dict = {}
        for k, v in image_dict.items():
            input_dict[k] = v
            if k == image_key:
                input_dict[k] = in_type(v)
        splitter = RandGridPatchd(keys=image_key, **input_parameters)
        splitter.set_random_state(1234)
        output = list(splitter(input_dict))
        self.assertEqual(len(output), len(expected))
        for output_patch, expected_patch in zip(output, expected):
            assert_allclose(output_patch[image_key], expected_patch, type_test=False)


if __name__ == "__main__":
    unittest.main()
