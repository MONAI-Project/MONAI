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

from collections.abc import Callable

from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.metrics import AveragePrecisionMetric
from monai.utils import Average


class AveragePrecision(IgniteMetricHandler):
    """
    Computes Average Precision (AP).
    accumulating predictions and the ground-truth during an epoch and applying `compute_average_precision`.

    Args:
        average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
            Type of averaging performed if not binary classification. Defaults to ``"macro"``.

            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.

        output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
            construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
            lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
            `engine.state` and `output_transform` inherit from the ignite concept:
            https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.

    Note:
        Average Precision expects y to be comprised of 0's and 1's.
        y_pred must either be probability estimates or confidence values.

    """

    def __init__(self, average: Average | str = Average.MACRO, output_transform: Callable = lambda x: x) -> None:
        metric_fn = AveragePrecisionMetric(average=Average(average))
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=False)
