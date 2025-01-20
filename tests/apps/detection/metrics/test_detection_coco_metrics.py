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

import random
import unittest

import numpy as np
import torch

from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data.box_utils import box_iou


class TestCOCOMetrics(unittest.TestCase):

    def test_coco_run(self):
        coco_metric = COCOMetric(classes=["c0", "c1", "c2"], iou_list=[0.1], max_detection=[10])

        num_images = 10

        val_outputs_all = []
        val_targets_all = []
        for _ in range(num_images):
            # randomly generate gt boxes and pred boxes
            num_gt_boxes = random.randint(1, 3)
            num_pred_boxes = random.randint(0, 3)

            box_start = torch.randint(3, (num_pred_boxes, 3))
            box_stop = box_start + torch.randint(1, 32, (num_pred_boxes, 3))
            boxes = torch.cat((box_start, box_stop), dim=1).to(torch.float16)
            val_outputs_all.append(
                {
                    "boxes": boxes,
                    "labels": torch.randint(3, (num_pred_boxes,)),
                    "scores": torch.randn((num_pred_boxes,)).absolute(),
                }
            )

            box_start = torch.randint(3, (num_gt_boxes, 3))
            box_stop = box_start + torch.randint(1, 32, (num_gt_boxes, 3))
            boxes = torch.cat((box_start, box_stop), dim=1).to(torch.float16)
            val_targets_all.append({"boxes": boxes, "labels": torch.randint(3, (num_gt_boxes,))})

        results_metric = matching_batch(
            iou_fn=box_iou,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=[val_data_i["boxes"].numpy() for val_data_i in val_outputs_all],
            pred_classes=[val_data_i["labels"].numpy() for val_data_i in val_outputs_all],
            pred_scores=[val_data_i["scores"].numpy() for val_data_i in val_outputs_all],
            gt_boxes=[val_data_i["boxes"].numpy() for val_data_i in val_targets_all],
            gt_classes=[val_data_i["labels"].numpy() for val_data_i in val_targets_all],
        )
        val_epoch_metric_dict = coco_metric(results_metric)[0]
        np.testing.assert_array_less([-16.01], [sum(val_epoch_metric_dict.values())])


if __name__ == "__main__":
    unittest.main()
