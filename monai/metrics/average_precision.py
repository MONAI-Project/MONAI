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

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import torch

from monai.utils import Average, look_up_option

from .metric import CumulativeIterationMetric


class AveragePrecisionMetric(CumulativeIterationMetric):
    """
    Computes Average Precision (AP). Referring to: `sklearn.metrics.average_precision_score
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score>`_.
    The input `y_pred` and `y` can be a list of `channel-first` Tensor or a `batch-first` Tensor.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
            Type of averaging performed if not binary classification.
            Defaults to ``"macro"``.

            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.

    """

    def __init__(self, average: Average | str = Average.MACRO) -> None:
        super().__init__()
        self.average = average

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return y_pred, y

    def aggregate(self, average: Average | str | None = None) -> np.ndarray | float | npt.ArrayLike:
        """
        Typically `y_pred` and `y` are stored in the cumulative buffers at each iteration,
        This function reads the buffers and computes the Average Precision.

        Args:
            average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
                Type of averaging performed if not binary classification. Defaults to `self.average`.

        """
        y_pred, y = self.get_buffer()
        # compute final value and do metric reduction
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")

        return compute_average_precision(y_pred=y_pred, y=y, average=average or self.average)


def _calculate(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    if not (y.ndimension() == y_pred.ndimension() == 1 and len(y) == len(y_pred)):
        raise AssertionError("y and y_pred must be 1 dimension data with same length.")
    y_unique = y.unique()
    if len(y_unique) == 1:
        warnings.warn(f"y values can not be all {y_unique.item()}, skip AP computation and return `Nan`.")
        return float("nan")
    if not y_unique.equal(torch.tensor([0, 1], dtype=y.dtype, device=y.device)):
        warnings.warn(f"y values must be 0 or 1, but in {y_unique.tolist()}, skip AP computation and return `Nan`.")
        return float("nan")

    n = len(y)
    indices = y_pred.argsort(descending=True)
    y = y[indices].cpu().numpy()  # type: ignore[assignment]
    y_pred = y_pred[indices].cpu().numpy()  # type: ignore[assignment]
    npos = ap = tmp_pos = 0.0

    for i in range(n):
        y_i = cast(float, y[i])
        if i + 1 < n and y_pred[i] == y_pred[i + 1]:
            tmp_pos += y_i
        else:
            tmp_pos += y_i
            npos += tmp_pos
            ap += tmp_pos * npos / (i + 1)
            tmp_pos = 0

    return ap / npos


def compute_average_precision(
    y_pred: torch.Tensor, y: torch.Tensor, average: Average | str = Average.MACRO
) -> np.ndarray | float | npt.ArrayLike:
    """Computes Average Precision (AP). Referring to: `sklearn.metrics.average_precision_score
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score>`_.

    Args:
        y_pred: input data to compute, typical classification model output.
            the first dim must be batch, if multi-classes, it must be in One-Hot format.
            for example: shape `[16]` or `[16, 1]` for a binary data, shape `[16, 2]` for 2 classes data.
        y: ground truth to compute AP metric, the first dim must be batch.
            if multi-classes, it must be in One-Hot format.
            for example: shape `[16]` or `[16, 1]` for a binary data, shape `[16, 2]` for 2 classes data.
        average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
            Type of averaging performed if not binary classification.
            Defaults to ``"macro"``.

            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.

    Raises:
        ValueError: When ``y_pred`` dimension is not one of [1, 2].
        ValueError: When ``y`` dimension is not one of [1, 2].
        ValueError: When ``average`` is not one of ["macro", "weighted", "micro", "none"].

    Note:
        Average Precision expects y to be comprised of 0's and 1's. `y_pred` must be either prob. estimates or confidence values.

    """
    y_pred_ndim = y_pred.ndimension()
    y_ndim = y.ndimension()
    if y_pred_ndim not in (1, 2):
        raise ValueError(
            f"Predictions should be of shape (batch_size, num_classes) or (batch_size, ), got {y_pred.shape}."
        )
    if y_ndim not in (1, 2):
        raise ValueError(f"Targets should be of shape (batch_size, num_classes) or (batch_size, ), got {y.shape}.")
    if y_pred_ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(dim=-1)
        y_pred_ndim = 1
    if y_ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(dim=-1)

    if y_pred_ndim == 1:
        return _calculate(y_pred, y)

    if y.shape != y_pred.shape:
        raise ValueError(f"data shapes of y_pred and y do not match, got {y_pred.shape} and {y.shape}.")

    average = look_up_option(average, Average)
    if average == Average.MICRO:
        return _calculate(y_pred.flatten(), y.flatten())
    y, y_pred = y.transpose(0, 1), y_pred.transpose(0, 1)
    ap_values = [_calculate(y_pred_, y_) for y_pred_, y_ in zip(y_pred, y)]
    if average == Average.NONE:
        return ap_values
    if average == Average.MACRO:
        return np.mean(ap_values)
    if average == Average.WEIGHTED:
        weights = [sum(y_) for y_ in y]
        return np.average(ap_values, weights=weights)  # type: ignore[no-any-return]
    raise ValueError(f'Unsupported average: {average}, available options are ["macro", "weighted", "micro", "none"].')
