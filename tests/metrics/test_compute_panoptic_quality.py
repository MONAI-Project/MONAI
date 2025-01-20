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

from monai.metrics import PanopticQualityMetric, compute_panoptic_quality
from tests.util import SkipIfNoModule

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# TEST_FUNC_CASE related cases are used to test for single image with HW input shape

sample_1 = torch.randint(low=0, high=5, size=(64, 64), device=_device)
sample_2_pred = torch.as_tensor([[0, 1, 1, 1], [0, 0, 0, 0], [2, 0, 3, 3], [4, 2, 2, 0]], device=_device)
sample_2_pred_need_remap = torch.as_tensor([[0, 7, 7, 7], [0, 0, 0, 0], [1, 0, 8, 8], [9, 1, 1, 0]], device=_device)
sample_2_gt = torch.as_tensor([[1, 1, 2, 1], [0, 0, 0, 0], [1, 3, 0, 0], [4, 3, 3, 3]], device=_device)
# if pred == gt, result should be 1
TEST_FUNC_CASE_1 = [{"pred": sample_1, "gt": sample_1, "match_iou_threshold": 0.99}, 1.0]

# test sample_2 when match_iou_threshold = 0.5
TEST_FUNC_CASE_2 = [{"pred": sample_2_pred, "gt": sample_2_gt, "match_iou_threshold": 0.5}, 0.25]
# test sample_2 when match_iou_threshold = 0.3, metric_name = "sq"
TEST_FUNC_CASE_3 = [{"pred": sample_2_pred, "gt": sample_2_gt, "metric_name": "sq", "match_iou_threshold": 0.3}, 0.6]
# test sample_2 when match_iou_threshold = 0.3, pred has different order, metric_name = "RQ"
TEST_FUNC_CASE_4 = [
    {"pred": sample_2_pred_need_remap, "gt": sample_2_gt, "metric_name": "RQ", "match_iou_threshold": 0.3},
    0.75,
]

# TEST_CLS_CASE related cases are used to test the PanopticQualityMetric with B2HW input
sample_3_pred = torch.as_tensor(
    [
        [[[2, 0, 1], [2, 1, 1], [0, 1, 1]], [[0, 1, 3], [0, 0, 0], [1, 2, 1]]],
        [[[1, 1, 1], [3, 2, 0], [3, 2, 1]], [[1, 1, 3], [3, 1, 1], [0, 3, 0]]],
    ],
    device=_device,
)

sample_3_gt = torch.as_tensor(
    [
        [[[2, 0, 0], [2, 0, 0], [2, 2, 3]], [[3, 3, 3], [3, 2, 1], [2, 2, 3]]],
        [[[1, 1, 1], [0, 0, 3], [0, 0, 3]], [[0, 1, 3], [2, 1, 0], [3, 0, 3]]],
    ],
    device=_device,
)

# test sample_3, num_classes = 3, match_iou_threshold = 0.5
TEST_CLS_CASE_1 = [{"num_classes": 3, "match_iou_threshold": 0.5}, sample_3_pred, sample_3_gt, (0.0, 0.0, 0.25)]

# test sample_3, num_classes = 3, match_iou_threshold = 0.3
TEST_CLS_CASE_2 = [{"num_classes": 3, "match_iou_threshold": 0.3}, sample_3_pred, sample_3_gt, (0.25, 0.5, 0.25)]

# test sample_3, num_classes = 4, match_iou_threshold = 0.3, metric_name = "segmentation_quality"
TEST_CLS_CASE_3 = [
    {"num_classes": 4, "match_iou_threshold": 0.3, "metric_name": "segmentation_quality"},
    sample_3_pred,
    sample_3_gt,
    (0.5, 0.5, 1.0, 0.0),
]

# test sample_3, num_classes = 3, match_iou_threshold = 0.4, reduction = "none", metric_name = "Recognition Quality"
TEST_CLS_CASE_4 = [
    {"num_classes": 3, "reduction": "none", "match_iou_threshold": 0.4, "metric_name": "Recognition Quality"},
    sample_3_pred,
    sample_3_gt,
    [[0.0, 1.0, 0.0], [0.6667, 0.0, 0.4]],
]

# test sample_3, num_classes = 3, match_iou_threshold = 0.4, reduction = "none", multiple metrics
TEST_CLS_CASE_5 = [
    {"num_classes": 3, "reduction": "none", "match_iou_threshold": 0.4, "metric_name": ["Recognition Quality", "pq"]},
    sample_3_pred,
    sample_3_gt,
    [torch.as_tensor([[0.0, 1.0, 0.0], [0.6667, 0.0, 0.4]]), torch.as_tensor([[0.0, 0.5, 0.0], [0.3333, 0.0, 0.4]])],
]


@SkipIfNoModule("scipy.optimize")
class TestPanopticQualityMetric(unittest.TestCase):
    @parameterized.expand([TEST_FUNC_CASE_1, TEST_FUNC_CASE_2, TEST_FUNC_CASE_3, TEST_FUNC_CASE_4])
    def test_value(self, input_params, expected_value):
        result = compute_panoptic_quality(**input_params)
        np.testing.assert_allclose(result.cpu().detach().item(), expected_value, atol=1e-4)
        np.testing.assert_equal(result.device, input_params["pred"].device)

    @parameterized.expand([TEST_CLS_CASE_1, TEST_CLS_CASE_2, TEST_CLS_CASE_3, TEST_CLS_CASE_4, TEST_CLS_CASE_5])
    def test_value_class(self, input_params, y_pred, y_gt, expected_value):
        metric = PanopticQualityMetric(**input_params)
        metric(y_pred, y_gt)
        outputs = metric.aggregate()
        if isinstance(outputs, list):
            for output, value in zip(outputs, expected_value):
                np.testing.assert_allclose(output.cpu().numpy(), np.asarray(value), atol=1e-4)
        else:
            np.testing.assert_allclose(outputs.cpu().numpy(), np.asarray(expected_value), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
