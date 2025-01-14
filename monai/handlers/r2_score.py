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
from monai.metrics import R2Metric
from monai.utils import MultiOutput


class R2Score(IgniteMetricHandler):
    """
    Computes :math:`R^{2}` score accumulating predictions and the ground-truth during an epoch and applying `compute_r2_score`.

    Args:
        multi_output: {``"raw_values"``, ``"uniform_average"``, ``"variance_weighted"``}
            Type of aggregation performed on multi-output scores.
            Defaults to ``"uniform_average"``.

            - ``"raw_values"``: the scores for each output are returned.
            - ``"uniform_average"``: the scores of all outputs are averaged with uniform weight.
            - ``"variance_weighted"``: the scores of all outputs are averaged, weighted by the variances
              of each individual output.
        p: non-negative integer.
            Number of independent variables used for regression. ``p`` is used to compute adjusted :math:`R^{2}` score.
            Defaults to 0 (standard :math:`R^{2}` score).
        output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
            construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
            lists of `channel-first` Tensors. The form of `(y_pred, y)` is required by the `update()`.
            `engine.state` and `output_transform` inherit from the ignite concept:
            https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.

    See also:
        :py:class:`monai.metrics.R2Metric`

    """

    def __init__(
        self,
        multi_output: MultiOutput | str = MultiOutput.UNIFORM,
        p: int = 0,
        output_transform: Callable = lambda x: x,
    ) -> None:
        metric_fn = R2Metric(multi_output=multi_output, p=p)
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=False)
