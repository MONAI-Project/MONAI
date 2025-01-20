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

from monai.apps.detection.utils.ATSS_matcher import ATSSMatcher
from monai.data.box_utils import box_iou
from tests.util import assert_allclose

TEST_CASES = [
    [
        {"num_candidates": 2, "similarity_fn": box_iou, "center_in_gt": False},
        torch.tensor([[0, 1, 2, 3, 2, 5]], dtype=torch.float16),
        torch.tensor([[0, 1, 2, 3, 2, 5], [0, 1, 1, 3, 2, 5], [0, 1, 2, 3, 2, 4]], dtype=torch.float16),
        [3],
        3,
        torch.tensor([0, -1, -1]),
    ]
]


class TestATSS(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_atss(self, input_param, boxes, anchors, num_anchors_per_level, num_anchors_per_loc, expected_matches):
        matcher = ATSSMatcher(**input_param, debug=True)
        match_quality_matrix, matches = matcher.compute_matches(
            boxes, anchors, num_anchors_per_level, num_anchors_per_loc
        )
        assert_allclose(matches, expected_matches, type_test=True, device_test=True, atol=0)


if __name__ == "__main__":
    unittest.main()
