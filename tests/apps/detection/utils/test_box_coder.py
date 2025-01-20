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

from monai.apps.detection.utils.box_coder import BoxCoder
from monai.transforms import CastToType
from tests.util import assert_allclose


class TestBoxTransform(unittest.TestCase):
    def test_value(self):
        box_coder = BoxCoder(weights=[1, 1, 1, 1, 1, 1])
        test_dtype = [torch.float32, torch.float16]
        for dtype in test_dtype:
            gt_boxes_0 = torch.rand((10, 3)).abs()
            gt_boxes_1 = gt_boxes_0 + torch.rand((10, 3)).abs() + 10
            gt_boxes = torch.cat((gt_boxes_0, gt_boxes_1), dim=1)
            gt_boxes = CastToType(dtype=dtype)(gt_boxes)

            proposals_0 = (gt_boxes_0 + torch.rand(gt_boxes_0.shape)).abs()
            proposals_1 = proposals_0 + torch.rand(gt_boxes_0.shape).abs() + 10
            proposals = torch.cat((proposals_0, proposals_1), dim=1)

            rel_gt_boxes = box_coder.encode_single(gt_boxes, proposals)
            gt_back = box_coder.decode_single(rel_gt_boxes, proposals)
            assert_allclose(gt_back, gt_boxes, type_test=True, device_test=True, atol=0.1)


if __name__ == "__main__":
    unittest.main()
