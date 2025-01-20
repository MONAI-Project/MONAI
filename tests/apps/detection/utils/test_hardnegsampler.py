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

from monai.apps.detection.utils.hard_negative_sampler import HardNegativeSampler
from tests.util import assert_allclose

TEST_CASE = [
    [[], [], [], [torch.tensor([]), torch.tensor([])], [torch.tensor([]), torch.tensor([])]],
    [
        [0, 1],
        [1, 0, 2, 3],
        [0.1, 0.9, 0.4, 0.3, 0.3, 0.5],
        [torch.tensor([0, 1]), torch.tensor([1, 0, 1, 1])],
        [torch.tensor([1, 0]), torch.tensor([0, 1, 0, 0])],
    ],
]

select_sample_size_per_image = 6
positive_fraction = 0.5
min_neg = 1
pool_size = 2


class TestSampleSlices(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_shape(self, target_label0, target_label1, concat_fg_probs, expected_result_pos, expected_result_neg):
        compute_dtypes = [torch.float16, torch.float32]
        for compute_dtype in compute_dtypes:
            sampler = HardNegativeSampler(select_sample_size_per_image, positive_fraction, min_neg, pool_size)
            target_labels = [torch.tensor(target_label0), torch.tensor(target_label1)]
            result_pos, result_neg = sampler(target_labels, torch.tensor(concat_fg_probs, dtype=compute_dtype))
            for r, er in zip(result_pos, expected_result_pos):
                assert_allclose(r, er)
            for r, er in zip(result_neg, expected_result_neg):
                assert_allclose(r, er)


if __name__ == "__main__":
    unittest.main()
