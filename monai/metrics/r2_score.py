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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import torch

from monai.utils import MultiOutput, look_up_option

from .metric import CumulativeIterationMetric


class R2Metric(CumulativeIterationMetric):
    r"""Computes :math:`R^{2}` score (coefficient of determination):

    .. math::
        \operatorname {R^{2}}\left(Y, \hat{Y}\right) = 1 - \frac {\sum _{i=1}^{n}\left(y_i-\hat{y_i} \right)^{2}}
        {\sum _{i=1}^{n}\left(y_i-\bar{y} \right)^{2}},

    where :math:`\bar{y}` is the mean of observed :math:`y` ; or adjusted :math:`R^{2}` score:

    .. math::
        \operatorname {\bar{R}^{2}} = 1 - (1-R^{2}) \frac {n-1}{n-p-1},

    where :math:`p` is the number of independant variables used for the regression.

    More info: https://en.wikipedia.org/wiki/Coefficient_of_determination

    Input `y_pred` is compared with ground truth `y`.
    `y_pred` and `y` are expected to be 1D (single-output regression) or 2D (multi-output regression) real-valued
    tensors of same shape.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        multi_output: {``"raw_values"``, ``"uniform_average"``, ``"variance_weighted"``}
            Type of aggregation performed on multi-output scores.
            Defaults to ``"uniform_average"``.

            - ``"raw_values"``: the scores for each output are returned.
            - ``"uniform_average"``: the scores of all outputs are averaged with uniform weight.
            - ``"variance_weighted"``: the scores of all outputs are averaged, weighted by the variances of
              each individual output.
        p: non-negative integer.
            Number of independent variables used for regression. ``p`` is used to compute adjusted :math:`R^{2}` score.
            Defaults to 0 (standard :math:`R^{2}` score).

    """

    def __init__(self, multi_output: MultiOutput | str = MultiOutput.UNIFORM, p: int = 0) -> None:
        super().__init__()
        multi_output, p = _check_r2_params(multi_output, p)
        self.multi_output = multi_output
        self.p = p

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        _check_dim(y_pred, y)
        return y_pred, y

    def aggregate(self, multi_output: MultiOutput | str | None = None) -> np.ndarray | float | npt.ArrayLike:
        """
        Typically `y_pred` and `y` are stored in the cumulative buffers at each iteration,
        This function reads the buffers and computes the :math:`R^{2}` score.

        Args:
            multi_output: {``"raw_values"``, ``"uniform_average"``, ``"variance_weighted"``}
                Type of aggregation performed on multi-output scores. Defaults to `self.multi_output`.

        """
        y_pred, y = self.get_buffer()
        return compute_r2_score(y_pred=y_pred, y=y, multi_output=multi_output or self.multi_output, p=self.p)


def _check_dim(y_pred: torch.Tensor, y: torch.Tensor) -> None:
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y.shape != y_pred.shape:
        raise ValueError(f"data shapes of y_pred and y do not match, got {y_pred.shape} and {y.shape}.")

    dim = y.ndimension()
    if dim not in (1, 2):
        raise ValueError(
            f"predictions and ground truths should be of shape (batch_size, num_outputs) or (batch_size, ), got {y.shape}."
        )


def _check_r2_params(multi_output: MultiOutput | str, p: int) -> tuple[MultiOutput | str, int]:
    multi_output = look_up_option(multi_output, MultiOutput)
    if not isinstance(p, int) or p < 0:
        raise ValueError(f"`p` must be an integer larger or equal to 0, got {p}.")

    return multi_output, p


def _calculate(y_pred: np.ndarray, y: np.ndarray, p: int) -> float:
    num_obs = len(y)
    rss = np.sum((y_pred - y) ** 2)
    tss = np.sum(y**2) - np.sum(y) ** 2 / num_obs
    r2 = 1 - (rss / tss)
    r2_adjusted = 1 - (1 - r2) * (num_obs - 1) / (num_obs - p - 1)

    return r2_adjusted  # type: ignore[no-any-return]


def compute_r2_score(
    y_pred: torch.Tensor, y: torch.Tensor, multi_output: MultiOutput | str = MultiOutput.UNIFORM, p: int = 0
) -> np.ndarray | float | npt.ArrayLike:
    """Computes :math:`R^{2}` score (coefficient of determination).

    Args:
        y_pred: input data to compute :math:`R^{2}` score, the first dim must be batch.
            For example: shape `[16]` or `[16, 1]` for a single-output regression, shape `[16, x]` for x output variables.
        y: ground truth to compute :math:`R^{2}` score, the first dim must be batch.
            For example: shape `[16]` or `[16, 1]` for a single-output regression, shape `[16, x]` for x output variables.
        multi_output: {``"raw_values"``, ``"uniform_average"``, ``"variance_weighted"``}
            Type of aggregation performed on multi-output scores.
            Defaults to ``"uniform_average"``.

            - ``"raw_values"``: the scores for each output are returned.
            - ``"uniform_average"``: the scores of all outputs are averaged with uniform weight.
            - ``"variance_weighted"``: the scores of all outputs are averaged, weighted by the variances
              each individual output.
        p: non-negative integer.
            Number of independent variables used for regression. ``p`` is used to compute adjusted :math:`R^{2}` score.
            Defaults to 0 (standard :math:`R^{2}` score).

    Raises:
        ValueError: When ``multi_output`` is not one of ["raw_values", "uniform_average", "variance_weighted"].
        ValueError: When ``p`` is not a non-negative integer.
        ValueError: When ``y_pred`` or ``y`` are not PyTorch tensors.
        ValueError: When ``y_pred`` and ``y`` don't have the same shape.
        ValueError: When ``y_pred`` or ``y`` dimension is not one of [1, 2].
        ValueError: When n_samples is less than 2.
        ValueError: When ``p`` is greater or equal to n_samples - 1.

    """
    multi_output, p = _check_r2_params(multi_output, p)
    _check_dim(y_pred, y)
    dim = y.ndimension()
    n = y.shape[0]
    y = y.cpu().numpy()  # type: ignore[assignment]
    y_pred = y_pred.cpu().numpy()  # type: ignore[assignment]

    if n < 2:
        raise ValueError("There is no enough data for computing. Needs at least two samples to calculate r2 score.")
    if p >= n - 1:
        raise ValueError("`p` must be smaller than n_samples - 1, " f"got p={p}, n_samples={n}.")

    if dim == 2 and y_pred.shape[1] == 1:
        y_pred = np.squeeze(y_pred, axis=-1)  # type: ignore[assignment]
        y = np.squeeze(y, axis=-1)  # type: ignore[assignment]
        dim = 1

    if dim == 1:
        return _calculate(y_pred, y, p)  # type: ignore[arg-type]

    y, y_pred = np.transpose(y, axes=(1, 0)), np.transpose(y_pred, axes=(1, 0))  # type: ignore[assignment]
    r2_values = [_calculate(y_pred_, y_, p) for y_pred_, y_ in zip(y_pred, y)]
    if multi_output == MultiOutput.RAW:
        return r2_values
    if multi_output == MultiOutput.UNIFORM:
        return np.mean(r2_values)
    if multi_output == MultiOutput.VARIANCE:
        weights = np.var(y, axis=1)
        return np.average(r2_values, weights=weights)  # type: ignore[no-any-return]
    raise ValueError(
        f'Unsupported multi_output: {multi_output}, available options are ["raw_values", "uniform_average", "variance_weighted"].'
    )
