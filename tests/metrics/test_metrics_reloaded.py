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

from monai.metrics import MetricsReloadedBinary, MetricsReloadedCategorical
from monai.utils import optional_import

_, has_metrics = optional_import("MetricsReloaded")

# shape: (1, 1, 2, 2)
y_pred = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
y = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
TEST_CASES_BINARY = [
    [{"metric_name": "False Positives"}, [y_pred, y], 0.0],
    [{"metric_name": "False Negatives"}, [y_pred, y], 1.0],
    [{"metric_name": "True Positives"}, [y_pred, y], 2.0],
    [{"metric_name": "True Negatives"}, [y_pred, y], 1.0],
    [{"metric_name": "Youden Index"}, [y_pred, y], 0.666654],
    [{"metric_name": "Sensitivity"}, [y_pred, y], 0.666664],
    [{"metric_name": "Specificity"}, [y_pred, y], 0.99999],
    [{"metric_name": "Balanced Accuracy"}, [y_pred, y], 0.833327],
    [{"metric_name": "Accuracy"}, [y_pred, y], 0.75],
    [{"metric_name": "False Positive Rate"}, [y_pred, y], 0.0],
    [{"metric_name": "Normalised Expected Cost"}, [y_pred, y], 0.333333],
    [{"metric_name": "Matthews Correlation Coefficient"}, [y_pred, y], 0.57735],
    [{"metric_name": "Cohens Kappa"}, [y_pred, y], 0.5],
    [{"metric_name": "Positive Likelihood Ratio"}, [y_pred, y], 66576.03],
    [{"metric_name": "Prediction Overlaps Reference"}, [y_pred, y], 1.0],
    [{"metric_name": "Positive Predictive Value"}, [y_pred, y], 0.999995],
    [{"metric_name": "Recall"}, [y_pred, y], 0.666664],
    [{"metric_name": "FBeta"}, [y_pred, y], 0.799992],
    [{"metric_name": "Net Benefit Treated"}, [y_pred, y], 0.5],
    [{"metric_name": "Negative Predictive Values"}, [y_pred, y], 0.5],
    [{"metric_name": "Dice Score"}, [y_pred, y], 0.799992],
    [{"metric_name": "False Positives Per Image"}, [y_pred, y], 0.0],
    [{"metric_name": "Intersection Over Reference"}, [y_pred, y], 0.666664],
    [{"metric_name": "Intersection Over Union"}, [y_pred, y], 0.666664],
    [{"metric_name": "Volume Difference"}, [y_pred, y], 0.333333],
    [{"metric_name": "Topology Precision"}, [y_pred, y], 1.0],
    [{"metric_name": "Topology Sensitivity"}, [y_pred, y], 1.0],
    [{"metric_name": "Centreline Dice Score"}, [y_pred, y], 1.0],
    [{"metric_name": "Boundary IoU"}, [y_pred, y], 0.666667],
    [{"metric_name": "Normalised Surface Distance"}, [y_pred, y], 1.0],
    [{"metric_name": "Average Symmetric Surface Distance"}, [y_pred, y], 0.2],
    [{"metric_name": "Mean Average Surfance Distance"}, [y_pred, y], 0.166666],
    [{"metric_name": "Hausdorff Distance"}, [y_pred, y], 1.0],
    [{"metric_name": "xTh Percentile Hausdorff Distance"}, [y_pred, y], 0.9],
]

# shape: (1, 3, 2, 2)
y_pred = torch.tensor([[[[0, 0], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 0]]]])
y = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]]])
TEST_CASES_CATEGORICAL = [
    [{"metric_name": "Balanced Accuracy"}, [y_pred, y], 0.5],
    [{"metric_name": "Weighted Cohens Kappa"}, [y_pred, y], 0.272727],
    [{"metric_name": "Matthews Correlation Coefficient"}, [y_pred, y], 0.387298],
    [{"metric_name": "Expected Cost"}, [y_pred, y], 0.5],
    [{"metric_name": "Normalised Expected Cost"}, [y_pred, y], 0.75],
]


@unittest.skipIf(not has_metrics, "MetricsReloaded not available.")
class TestMetricsReloaded(unittest.TestCase):

    @parameterized.expand(TEST_CASES_BINARY)
    def test_binary(self, input_param, input_data, expected_val):
        metric = MetricsReloadedBinary(**input_param)
        result = metric(*input_data)
        np.testing.assert_allclose(
            result.detach().cpu().numpy(), expected_val, rtol=1e-5, err_msg=input_param["metric_name"]
        )

    @parameterized.expand(TEST_CASES_CATEGORICAL)
    def test_categorical(self, input_param, input_data, expected_val):
        metric = MetricsReloadedCategorical(**input_param)
        result = metric(*input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
