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
from collections.abc import Sequence

import numpy as np
import torch

from monai.metrics.utils import (
    do_metric_reduction,
    get_mask_edges,
    get_surface_distance,
    ignore_background,
    prepare_spacing,
)
from monai.utils import MetricReduction, convert_data_type

from .metric import CumulativeIterationMetric

__all__ = ["HausdorffDistanceMetric", "compute_hausdorff_distance", "compute_percent_hausdorff_distance"]


class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: float | None = None,
        directed: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[float]] | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.
            spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
                If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
                the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
                If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
                If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
                else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
                for all images in batch. Defaults to ``None``.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
            spacing=spacing,
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_hausdorff_distance(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    directed: bool = False,
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[float]] | None = None,
) -> torch.Tensor:
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
            the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
            If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
            If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
            else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
            for all images in batch. Defaults to ``None``.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    hd = np.empty((batch_size, n_class))

    img_dim = y_pred.ndim - 2
    spacing = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan/inf distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan/inf distance.")

        distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile, spacing[b])
        if directed:
            hd[b, c] = distance_1
        else:
            distance_2 = compute_percent_hausdorff_distance(
                edges_gt, edges_pred, distance_metric, percentile, spacing[b]
            )
            hd[b, c] = max(distance_1, distance_2)
    return convert_data_type(hd, output_type=torch.Tensor, device=y_pred.device, dtype=torch.float)[0]


def compute_percent_hausdorff_distance(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    spacing: int | float | list | np.ndarray | None = None,
) -> float:
    """
    This function is used to compute the directed Hausdorff distance.
    """

    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric=distance_metric, spacing=spacing)

    # for both pred and gt do not have foreground
    if surface_distance.shape == (0,):
        return np.nan

    if not percentile:
        return surface_distance.max()  # type: ignore[no-any-return]

    if 0 <= percentile <= 100:
        return np.percentile(surface_distance, percentile)  # type: ignore[no-any-return]
    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
