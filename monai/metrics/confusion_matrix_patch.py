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

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, ensure_tuple

from .metric import CumulativeIterationMetric


class ConfusionMatrixMetricPatch(CumulativeIterationMetric):
    """
    Compute confusion matrix related metrics. This function supports to calculate all metrics mentioned in:
    `Confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.
    It can support both multi-classes and multi-labels classification and segmentation tasks.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` for an instance to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include metric computation on the first channel of
            the predicted output. Defaults to True.
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
            Except for input only one metric, multiple metrics are also supported via input a sequence of metric names, such as
            ("sensitivity", "precision", "recall"), if ``compute_sample`` is ``True``, multiple ``f`` and ``not_nans`` will be
            returned with the same order as input names when calling the class.
        compute_sample: when reducing, if ``True``, each sample's metric will be computed based on each confusion matrix first.
            if ``False``, compute reduction on the confusion matrices first, defaults to ``False``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns [(metric, not_nans), ...]. If False,
            aggregate() returns [metric, ...].
            Here `not_nans` count the number of not nans for True Positive, False Positive, True Negative and False Negative.
            Its shape depends on the shape of the metric, and it has one more dimension with size 4. For example, if the shape
            of the metric is [3, 3], `not_nans` has the shape [3, 3, 4].

    """

    def __init__(
        self,
        include_background: bool = True,
        metric_name: Sequence[str] | str = "hit_rate",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute. It must be one-hot format and first dim is batch.
                The values should be binarized.
            y: ground truth to compute the metric. It must be one-hot format and first dim is batch.
                The values should be binarized.
        Raises:
            ValueError: when `y_pred` has less than two dimensions.
        """
        # check dimension
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False

        return get_confusion_matrix(y_pred=y_pred, y=y, include_background=self.include_background)

    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Execute reduction for the confusion matrix values.

        Args:
            compute_sample: when reducing, if ``True``, each sample's metric will be computed based on each confusion matrix first.
                if ``False``, compute reduction on the confusion matrices first, defaults to ``False``.
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            if compute_sample or self.compute_sample:
                sub_confusion_matrix = compute_confusion_matrix_metric(metric_name, data)
                f, not_nans = do_metric_reduction(sub_confusion_matrix, reduction or self.reduction)
            else:
                f, not_nans = do_metric_reduction(data, reduction or self.reduction)
                f = compute_confusion_matrix_metric(metric_name, f)
            if self.get_not_nans:
                results.append((f, not_nans))
            else:
                results.append(f)
        return results


def get_confusion_matrix(y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = True) -> torch.Tensor:
    """
    Compute confusion matrix. A tensor with the shape [BC4] will be returned. Where, the third dimension
    represents the number of true positive, false positive, true negative and false negative values for
    each channel of each sample within the input batch. Where, B equals to the batch size and C equals to
    the number of classes that need to be computed.

    Args:
        y_pred: input data to compute. It must be one-hot format and first dim is batch.
            The values should be binarized.
        y: ground truth to compute the metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include metric computation on the first channel of
            the predicted output. Defaults to True.

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # get confusion matrix related metric
    batch_size, n_class = y_pred.shape[:2]
    # convert to [BNS], where S is the number of pixels for one sample.
    # As for classification tasks, S equals to 1.
    y_pred = y_pred.reshape(batch_size, n_class, -1)
    y = y.reshape(batch_size, n_class, -1)
    tp = (y_pred + y) == 2
    tn = (y_pred + y) == 0

    tp = tp.sum(dim=[2]).float()
    tn = tn.sum(dim=[2]).float()
    p = y.sum(dim=[2]).float()
    n = y.shape[-1] - p

    fn = p - tp
    fp = n - tn

    return torch.stack([tp, fp, tn, fn], dim=-1)


"""
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
        confusion_matrix: Please see the doc string of the function ``get_confusion_matrix`` for more details.

    Raises:
        ValueError: when the size of the last dimension of confusion_matrix is not 4.
        NotImplementedError: when specify a not implemented metric_name.

    """

def compute_confusion_matrix_metric(metric_name: str, confusion_matrix: torch.Tensor) -> torch.Tensor:

    metric = check_confusion_matrix_metric_name(metric_name)

    """ Ckeck dimensionality of confusion_matrix tensor """
    input_dim = confusion_matrix.ndimension()

    """
    If confusion_matrix is a one-dimensional tensor, it will be given a new dimension.
    If the size of the last dimension of confusion_matrix is not 4 (the expected size of a standard 2x2 confusion matrix), a ValueError will be raised.
    """
    if input_dim == 1:
        confusion_matrix = confusion_matrix.unsqueeze(dim=0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError("the size of the last dimension of confusion_matrix should be 4.")

    tp = confusion_matrix[..., 0]   # get True Positive (TP) from confusion_metrix
    fp = confusion_matrix[..., 1]   # get False Positive (FP) ...
    tn = confusion_matrix[..., 2]   # get True Negative (TN) ...
    fn = confusion_matrix[..., 3]   # get False Negative (FN) ...
    p = tp + fn     # total number of actual positive cases
    n = fp + tn     # total number of actual negative cases

    # calculate metric
    numerator: torch.Tensor
    denominator: torch.Tensor | float
    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)

    """
    1. tpr - True Positive Rate (Recall): The ratio of correctly predicted positive samples to the total number of samples that are actually positive.
    2. tnr - True Negative Rate: The proportion of correctly predicted negative samples over the total number of samples that are actually negative.
    3. ppv - Positive Predictive Value (Precision): The ratio of correctly predicted positive samples to the total number of samples predicted to be positive.
    4. npv - Negative Predictive Value: The ratio of correctly predicted negative samples to the total number of samples predicted to be negative.
    5. fnr - False Negative Rate: The ratio of positive samples that are incorrectly predicted to be negative to the total number of samples that are actually positive.
    6. fpr - False Positive Rate: The ratio of negative samples that are incorrectly predicted as positive to the total number of samples that are actually negative.
    7. fdr - False Discovery Rate:  The ratio of predicted positive samples that are actually negative to the total number of samples predicted to be positive.
    8. for - False Omission Rate: The ratio of predicted negative samples that are actually positive to the total number of samples predicted to be negative.
    9. pt - Prevalence Threshold: It provides insight into the optimal balance point for deciding a positive or negative classification based on the prevalence of the condition in the dataset.
    10. ts - Threat Score: It measures the proportion of correct predictions among all relevant events.
    11. acc - Accuracy:  It measures the proportion of correctly classified instances out of the total number of instances in a dataset.
    12. ba - Balanced Accuracy: It adjusts the traditional accuracy by accounting for both the True Positive Rate (Sensitivity) and the True Negative Rate (Specificity).
    13. f1 - F1-score: It is a performance metric for classification tasks, especially useful when the dataset is imbalanced. It combines Precision and Recall into a single metric by calculating their harmonic mean.
    14. mcc - Matthews Correlation Coefficient: A more robust measure of correlation between prediction and observation than accuracy, especially in cases of imbalanced classes.
    15. fm - Fowlkes-Mallows Index: It measures the geometric mean of precision and recall.
    16. bm - Informedness: It measures the extent to which the model's predictions are better than random guessing.
    17. mk - Markedness: It measures the extent to which the model's predictions are better than random guessing, focusing on the positive class.
    """

    match metric:
        case "tpr":
            numerator, denominator = tp, p
        case "tnr": #
            numerator, denominator = tn, n
        case "ppv":
            numerator, denominator = tp, (tp + fp)
        case "npv":
            numerator, denominator = tn, (tn + fn)
        case "fnr":
            numerator, denominator = fn, p
        case "fpr":
            numerator, denominator = fp, n
        case "fdr":
            numerator, denominator = fp, (fp + tp)
        case "for":
            numerator, denominator = fn, (fn + tn)
        case "pt":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            tnr = torch.where(n > 0, tn / n, nan_tensor)
            numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
            denominator = tpr + tnr - 1.0
        case "ts":
            numerator, denominator = tp, (tp + fn + fp)
        case "acc":
            numerator, denominator = (tp + tn), (p + n)
        case "ba":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            tnr = torch.where(n > 0, tn / n, nan_tensor)
            numerator, denominator = (tpr + tnr), 2.0
        case "f1":
            numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
        case "mcc":
            numerator = tp * tn - fp * fn
            denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        case "fm":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
            numerator = torch.sqrt(ppv * tpr)
            denominator = 1.0
        case "bm":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            tnr = torch.where(n > 0, tn / n, nan_tensor)
            numerator = tpr + tnr - 1.0
            denominator = 1.0
        case "mk":
            ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
            npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
            numerator = ppv + npv - 1.0
            denominator = 1.0
        case _:
            raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator


def check_confusion_matrix_metric_name(metric_name: str) -> str:
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in ["informedness", "bookmaker_informedness", "bm", "youden_index", "youden"]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    raise NotImplementedError("the metric is not implemented.")


from sklearn.metrics import confusion_matrix

y_test = [0 ,1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]

print(confusion_matrix(y_test, y_pred))
