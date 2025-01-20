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

from monai.apps.detection.utils.box_selector import BoxSelector
from tests.util import assert_allclose

device = "cuda" if torch.cuda.is_available() else "cpu"
num_anchors = 7

TEST_CASE = []
TEST_CASE.append(
    [  # 2D
        {
            "apply_sigmoid": False,
            "score_thresh": 0.1,
            "topk_candidates_per_level": 2,
            "nms_thresh": 0.1,
            "detections_per_img": 5,
        },
        [torch.tensor([[1, 2, 3, 2, 3, 4], [5, 6, 7, 8, 9, 10]]).to(torch.float16)],
        [torch.tensor([[0.1, 0.6], [0.2, 0.2]])],
        (20, 20, 20),
        torch.tensor([[1, 2, 3, 2, 3, 4], [5, 6, 7, 8, 9, 10]]),
    ]
)
TEST_CASE.append(
    [
        {
            "apply_sigmoid": False,
            "score_thresh": 0.1,
            "topk_candidates_per_level": 1,
            "nms_thresh": 0.1,
            "detections_per_img": 5,
        },
        [torch.tensor([[1, 2, 3, 2, 3, 4]]).to(torch.float32), torch.tensor([[5, 6, 7, 8, 9, 10]]).to(torch.float32)],
        [torch.tensor([[0.3, 0.6]]), torch.tensor([[0.2, 0.2]])],
        (20, 20, 8),
        torch.tensor([[1, 2, 3, 2, 3, 4], [5, 6, 7, 8, 9, 8]]),
    ]
)


class TestBoxSelector(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_box_selector(self, input_param, boxes, logits, image_shape, expected_results):
        box_selector = BoxSelector(**input_param)
        result = box_selector.select_boxes_per_image(boxes, logits, image_shape)
        assert_allclose(result[0], expected_results, type_test=True, device_test=False, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
