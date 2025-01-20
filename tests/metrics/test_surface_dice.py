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
import torch.nn.functional as F

from monai.metrics.surface_dice import SurfaceDiceMetric, compute_surface_dice
from tests.util import assert_allclose

_device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TestAllSurfaceDiceMetrics(unittest.TestCase):
    def test_tolerance_euclidean_distance_with_spacing(self):
        batch_size = 2
        n_class = 2
        test_spacing = (0.85, 1.2)
        predictions = torch.zeros((batch_size, 480, 640), dtype=torch.int64, device=_device)
        labels = torch.zeros((batch_size, 480, 640), dtype=torch.int64, device=_device)
        predictions[0, :, 50:] = 1
        labels[0, :, 60:] = 1  # 10 px shift
        predictions_hot = F.one_hot(predictions, num_classes=n_class).permute(0, 3, 1, 2)
        labels_hot = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2)

        sd0 = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True)
        res0 = sd0(predictions_hot, labels_hot, spacing=test_spacing)
        agg0 = sd0.aggregate()  # aggregation: nanmean across image then nanmean across batch
        sd0_nans = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True, get_not_nans=True)
        res0_nans = sd0_nans(predictions_hot, labels_hot)
        agg0_nans, not_nans = sd0_nans.aggregate()

        np.testing.assert_array_equal(res0.cpu(), res0_nans.cpu())
        np.testing.assert_equal(res0.device, predictions.device)
        np.testing.assert_array_equal(agg0.cpu(), agg0_nans.cpu())
        np.testing.assert_equal(agg0.device, predictions.device)

        res1 = SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(
            predictions_hot, labels_hot, spacing=test_spacing
        )
        res9 = SurfaceDiceMetric(class_thresholds=[9, 9], include_background=True)(
            predictions_hot, labels_hot, spacing=test_spacing
        )
        res10 = SurfaceDiceMetric(class_thresholds=[10, 10], include_background=True)(
            predictions_hot, labels_hot, spacing=test_spacing
        )
        res11 = SurfaceDiceMetric(class_thresholds=[11, 11], include_background=True)(
            predictions_hot, labels_hot, spacing=test_spacing
        )
        # because spacing is (0.85, 1.2) and we moved 10 pixels in the columns direction,
        # everything with tolerance 12 or more should be the same as tolerance 12 (surface dice is 1.0)
        res12 = SurfaceDiceMetric(class_thresholds=[12, 12], include_background=True)(
            predictions_hot, labels_hot, spacing=test_spacing
        )
        res13 = SurfaceDiceMetric(class_thresholds=[13, 13], include_background=True)(
            predictions_hot, labels_hot, spacing=test_spacing
        )

        for res in [res0, res9, res10, res11, res12, res13]:
            assert res.shape == torch.Size([2, 2])

        assert res0[0, 0] < res1[0, 0] < res9[0, 0] < res10[0, 0] < res11[0, 0]
        assert res0[0, 1] < res1[0, 1] < res9[0, 1] < res10[0, 1] < res11[0, 1]
        np.testing.assert_array_equal(res12.cpu(), res13.cpu())

        expected_res0 = np.zeros((batch_size, n_class))
        expected_res0[0, 1] = 1 - (478 + 480 + 9 * 2) / (480 * 4 + 588 * 2 + 578 * 2)
        expected_res0[0, 0] = 1 - (478 + 480 + 9 * 2) / (480 * 4 + 48 * 2 + 58 * 2)
        expected_res0[1, 0] = 1
        expected_res0[1, 1] = np.nan
        for b, c in np.ndindex(batch_size, n_class):
            np.testing.assert_allclose(expected_res0[b, c], res0[b, c].cpu())
        np.testing.assert_array_equal(agg0.cpu(), np.nanmean(np.nanmean(expected_res0, axis=1), axis=0))
        np.testing.assert_equal(not_nans.cpu(), torch.tensor(2))

    def test_tolerance_euclidean_distance(self):
        batch_size = 2
        n_class = 2
        predictions = torch.zeros((batch_size, 480, 640), dtype=torch.int64, device=_device)
        labels = torch.zeros((batch_size, 480, 640), dtype=torch.int64, device=_device)
        predictions[0, :, 50:] = 1
        labels[0, :, 60:] = 1  # 10 px shift
        predictions_hot = F.one_hot(predictions, num_classes=n_class).permute(0, 3, 1, 2)
        labels_hot = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2)

        sd0 = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True)
        res0 = sd0(predictions_hot, labels_hot)
        agg0 = sd0.aggregate()  # aggregation: nanmean across image then nanmean across batch
        sd0_nans = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True, get_not_nans=True)
        res0_nans = sd0_nans(predictions_hot, labels_hot)
        agg0_nans, not_nans = sd0_nans.aggregate()

        np.testing.assert_array_equal(res0.cpu(), res0_nans.cpu())
        np.testing.assert_equal(res0.device, predictions.device)
        np.testing.assert_array_equal(agg0.cpu(), agg0_nans.cpu())
        np.testing.assert_equal(agg0.device, predictions.device)

        res1 = SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions_hot, labels_hot)
        res9 = SurfaceDiceMetric(class_thresholds=[9, 9], include_background=True)(predictions_hot, labels_hot)
        res10 = SurfaceDiceMetric(class_thresholds=[10, 10], include_background=True)(predictions_hot, labels_hot)
        res11 = SurfaceDiceMetric(class_thresholds=[11, 11], include_background=True)(predictions_hot, labels_hot)

        for res in [res0, res9, res10, res11]:
            assert res.shape == torch.Size([2, 2])

        assert res0[0, 0] < res1[0, 0] < res9[0, 0] < res10[0, 0]
        assert res0[0, 1] < res1[0, 1] < res9[0, 1] < res10[0, 1]
        np.testing.assert_array_equal(res10.cpu(), res11.cpu())

        expected_res0 = np.zeros((batch_size, n_class))
        expected_res0[0, 1] = 1 - (478 + 480 + 9 * 2) / (480 * 4 + 588 * 2 + 578 * 2)
        expected_res0[0, 0] = 1 - (478 + 480 + 9 * 2) / (480 * 4 + 48 * 2 + 58 * 2)
        expected_res0[1, 0] = 1
        expected_res0[1, 1] = np.nan
        for b, c in np.ndindex(batch_size, n_class):
            np.testing.assert_allclose(expected_res0[b, c], res0[b, c].cpu())
        np.testing.assert_array_equal(agg0.cpu(), np.nanmean(np.nanmean(expected_res0, axis=1), axis=0))
        np.testing.assert_equal(not_nans.cpu(), torch.tensor(2))

    def test_tolerance_euclidean_distance_3d(self):
        batch_size = 2
        n_class = 2
        predictions = torch.zeros((batch_size, 200, 110, 80), dtype=torch.int64, device=_device)
        labels = torch.zeros((batch_size, 200, 110, 80), dtype=torch.int64, device=_device)
        predictions[0, :, :, 20:] = 1
        labels[0, :, :, 30:] = 1  # offset by 10
        predictions_hot = F.one_hot(predictions, num_classes=n_class).permute(0, 4, 1, 2, 3)
        labels_hot = F.one_hot(labels, num_classes=n_class).permute(0, 4, 1, 2, 3)

        sd0 = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True)
        res0 = sd0(predictions_hot, labels_hot)
        agg0 = sd0.aggregate()  # aggregation: nanmean across image then nanmean across batch
        sd0_nans = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True, get_not_nans=True)
        res0_nans = sd0_nans(predictions_hot, labels_hot)
        agg0_nans, not_nans = sd0_nans.aggregate()

        np.testing.assert_array_equal(res0.cpu(), res0_nans.cpu())
        np.testing.assert_equal(res0.device, predictions.device)
        np.testing.assert_array_equal(agg0.cpu(), agg0_nans.cpu())
        np.testing.assert_equal(agg0.device, predictions.device)

        res1 = SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions_hot, labels_hot)
        res10 = SurfaceDiceMetric(class_thresholds=[10, 10], include_background=True)(predictions_hot, labels_hot)
        res11 = SurfaceDiceMetric(class_thresholds=[11, 11], include_background=True)(predictions_hot, labels_hot)

        for res in [res0, res1, res10, res11]:
            assert res.shape == torch.Size([2, 2])

        assert res0[0, 0] < res1[0, 0] < res10[0, 0]
        assert res0[0, 1] < res1[0, 1] < res10[0, 1]
        np.testing.assert_array_equal(res10.cpu(), res11.cpu())

        expected_res0 = np.zeros((batch_size, n_class))
        expected_res0[0, 1] = 1 - (200 * 110 + 198 * 108 + 9 * 200 * 2 + 9 * 108 * 2) / (
            200 * 110 * 4 + (58 + 48) * 200 * 2 + (58 + 48) * 108 * 2
        )
        expected_res0[0, 0] = 1 - (200 * 110 + 198 * 108 + 9 * 200 * 2 + 9 * 108 * 2) / (
            200 * 110 * 4 + (28 + 18) * 200 * 2 + (28 + 18) * 108 * 2
        )
        expected_res0[1, 0] = 1
        expected_res0[1, 1] = np.nan
        for b, c in np.ndindex(batch_size, n_class):
            np.testing.assert_allclose(expected_res0[b, c], res0[b, c].cpu())
        np.testing.assert_array_equal(agg0.cpu(), np.nanmean(np.nanmean(expected_res0, axis=1), axis=0))
        np.testing.assert_equal(not_nans.cpu(), torch.tensor(2))

    def test_tolerance_all_distances(self):
        batch_size = 1
        n_class = 2
        predictions = torch.zeros((batch_size, 10, 10), dtype=torch.int64)
        labels = torch.zeros((batch_size, 10, 10), dtype=torch.int64)
        predictions[0, 1:4, 1] = 1
        """
        [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
        """
        labels[0, 5:8, 6] = 1
        """
        [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
        """
        predictions_hot = F.one_hot(predictions, num_classes=n_class).permute(0, 3, 1, 2)
        labels_hot = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2)

        # Euclidean distance:
        # background:
        # 36 boundary pixels have 0 distances; non-zero distances:
        # distances gt_pred: [3, np.sqrt(9+4), 2, 3, 2, 2, 2, 1]
        # distances pred_gt: [1, 2, 2, 1]
        # class 1:
        # distances gt_pred: [sqrt(25+4), sqrt(25+9), sqrt(25+16)] = [5.38516481, 5.83095189, 6.40312424]
        # distances pred_gt: [sqrt(25+16), sqrt(25+9), sqrt(25+4)] = [6.40312424, 5.83095189, 5.38516481]

        res = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True)(predictions_hot, labels_hot)
        expected_res = [[1 - (8 + 4) / (36 * 2 + 8 + 4), 0]]
        np.testing.assert_array_almost_equal(res, expected_res)

        res = SurfaceDiceMetric(class_thresholds=[2.8, 5.5], include_background=True)(predictions_hot, labels_hot)
        expected_res = [[1 - 3 / (36 * 2 + 8 + 4), 1 - (2 + 2) / (3 + 3)]]
        np.testing.assert_array_almost_equal(res, expected_res)

        res = SurfaceDiceMetric(class_thresholds=[3, 6], include_background=True)(predictions_hot, labels_hot)
        expected_res = [[1 - 1 / (36 * 2 + 8 + 4), 1 - 2 / (3 + 3)]]
        np.testing.assert_array_almost_equal(res, expected_res)

        # Chessboard distance:
        # background:
        # 36 boundary pixels have 0 distances; non-zero distances:
        # distances gt_pred: [max(3,0), max(3,2), max(2,0), max(3,3), max(2,0), max(0,2), max(2,0), max(0,1)] =
        # [3, 3, 2, 3, 2, 2, 2, 1]
        # distances pred_gt: [max(0,1), max(2,0), max(2,0), max(1,0)] = [1, 2, 2, 1]
        # class 1:
        # distances gt_pred: [max(5,2), max(5,3), max(5,4)] = [5, 5, 5]
        # distances pred_gt: [max(5,4), max(5,3), max(5,2)] = [5, 5, 5]

        res = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True, distance_metric="chessboard")(
            predictions_hot, labels_hot
        )
        expected_res = [[1 - (8 + 4) / (36 * 2 + 8 + 4), 0]]
        np.testing.assert_array_almost_equal(res, expected_res)

        res = SurfaceDiceMetric(class_thresholds=[1, 4.999], include_background=True, distance_metric="chessboard")(
            predictions_hot, labels_hot
        )
        expected_res = [[1 - (7 + 2) / (36 * 2 + 8 + 4), 0]]
        np.testing.assert_array_almost_equal(res, expected_res)

        res = SurfaceDiceMetric(class_thresholds=[2, 5], include_background=True, distance_metric="chessboard")(
            predictions_hot, labels_hot
        )
        expected_res = [[1 - 3 / (36 * 2 + 8 + 4), 1]]
        np.testing.assert_array_almost_equal(res, expected_res)

        # Taxicab distance (= Manhattan distance):
        # background:
        # 36 boundary pixels have 0 distances; non-zero distances:
        # distances gt_pred: [3+0, 4+0, 2+0, 0+3, 2+0, 0+2, 2+0, 0+1] = [3, 4, 2, 3, 2, 2, 2, 1]
        # distances pred_gt: [0+1, 2+0, 2+0, 1+0] = [1, 2, 2, 1]
        # class 1:
        # distances gt_pred: [5+2, 5+3, 5+4] = [7, 8, 9]
        # distances pred_gt: [5+4, 5+3, 5+2] = [9, 8, 7]

        res = SurfaceDiceMetric(class_thresholds=[0, 0], include_background=True, distance_metric="taxicab")(
            predictions_hot, labels_hot
        )
        expected_res = [[1 - (8 + 4) / (36 * 2 + 8 + 4), 0]]
        np.testing.assert_array_almost_equal(res, expected_res)

        res = SurfaceDiceMetric(class_thresholds=[1, 7], include_background=True, distance_metric="taxicab")(
            predictions_hot, labels_hot
        )
        expected_res = [[1 - (7 + 2) / (36 * 2 + 8 + 4), 1 - (2 + 2) / (3 + 3)]]
        np.testing.assert_array_almost_equal(res, expected_res)

        res = SurfaceDiceMetric(class_thresholds=[3, 9], include_background=True, distance_metric="taxicab")(
            predictions_hot, labels_hot
        )
        expected_res = [[1 - 1 / (36 * 2 + 8 + 4), 1]]
        np.testing.assert_array_almost_equal(res, expected_res)

    def test_asserts(self):
        batch_size = 1
        n_class = 2
        predictions = torch.zeros((batch_size, 80, 80), dtype=torch.int64)
        labels = torch.zeros((batch_size, 80, 80), dtype=torch.int64)
        predictions[0, 10:20, 10:20] = 1
        labels[0, 20:30, 20:30] = 1
        predictions_hot = F.one_hot(predictions, num_classes=n_class).permute(0, 3, 1, 2)
        labels_hot = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2)

        # no torch tensor
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions_hot.numpy(), labels_hot)
        self.assertEqual(
            "y_pred or y must be a list/tuple of `channel-first` Tensors or a `batch-first` Tensor.",
            str(context.exception),
        )
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions_hot, labels_hot.numpy())
        self.assertEqual("y_pred and y must be PyTorch Tensor.", str(context.exception))

        # wrong dimensions
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions, labels_hot)
        self.assertEqual("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].", str(context.exception))
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions_hot, labels)
        self.assertEqual("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].", str(context.exception))

        # mismatch of shape of input tensors
        input_bad_shape = torch.clone(predictions_hot)
        input_bad_shape = input_bad_shape[:, :, :, :50]

        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[1, 1], include_background=True)(predictions_hot, input_bad_shape)
        self.assertEqual(
            "y_pred and y should have same shape, but instead, shapes are torch.Size([1, 2, 80, 80]) (y_pred) and "
            "torch.Size([1, 2, 80, 50]) (y).",
            str(context.exception),
        )

        # wrong number of class thresholds
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[1, 1, 1], include_background=True)(predictions_hot, labels_hot)
        self.assertEqual("number of classes (2) does not match number of class thresholds (3).", str(context.exception))

        # inf and nan values in class thresholds
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[np.inf, 1], include_background=True)(predictions_hot, labels_hot)
        self.assertEqual("All class thresholds need to be finite.", str(context.exception))

        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[np.nan, 1], include_background=True)(predictions_hot, labels_hot)
            self.assertEqual("All class thresholds need to be finite.", str(context.exception))

        # negative values in class thresholds:
        with self.assertRaises(ValueError) as context:
            SurfaceDiceMetric(class_thresholds=[-0.22, 1], include_background=True)(predictions_hot, labels_hot)
        self.assertEqual("All class thresholds need to be >= 0.", str(context.exception))

    def test_not_predicted_not_present(self):
        # class is present in labels, but not in prediction -> nsd of 0 should be yielded for that class; class is
        # neither present on labels, nor prediction -> nan should be yielded
        batch_size = 1
        n_class = 4
        predictions = torch.zeros((batch_size, 80, 80), dtype=torch.int64)
        labels = torch.zeros((batch_size, 80, 80), dtype=torch.int64)
        predictions[0, 10:20, 10:20] = 1
        labels[0, 10:20, 10:20] = 2
        predictions_hot = F.one_hot(predictions, num_classes=n_class).permute(0, 3, 1, 2)
        labels_hot = F.one_hot(labels, num_classes=n_class).permute(0, 3, 1, 2)

        # with and without background class
        sur_metric_bgr = SurfaceDiceMetric(class_thresholds=[1, 1, 1, 1], include_background=True)
        sur_metric = SurfaceDiceMetric(class_thresholds=[1, 1, 1], include_background=False)

        # test per-class results
        res_bgr_classes = sur_metric_bgr(predictions_hot, labels_hot)
        np.testing.assert_array_equal(res_bgr_classes, [[1, 0, 0, np.nan]])
        res_classes = sur_metric(predictions_hot, labels_hot)
        np.testing.assert_array_equal(res_classes, [[0, 0, np.nan]])

        # test aggregation
        res_bgr = sur_metric_bgr.aggregate(reduction="mean")
        np.testing.assert_equal(res_bgr, torch.tensor([1 / 3], dtype=torch.float))
        res = sur_metric.aggregate()
        np.testing.assert_equal(res, torch.tensor([0], dtype=torch.float))

        predictions_empty = torch.zeros((2, 3, 1, 1))
        sur_metric_nans = SurfaceDiceMetric(class_thresholds=[1, 1, 1], include_background=True, get_not_nans=True)
        res_classes = sur_metric_nans(predictions_empty, predictions_empty)
        res, not_nans = sur_metric_nans.aggregate()
        np.testing.assert_array_equal(res_classes, [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
        np.testing.assert_equal(res, torch.tensor([0], dtype=torch.float))
        np.testing.assert_equal(not_nans, torch.tensor([0], dtype=torch.float))

    def test_compute_surface_dice_subvoxel(self):
        mask_gt, mask_pred = (torch.zeros(1, 1, 128, 128, 128), torch.zeros(1, 1, 128, 128, 128))
        mask_gt[0, 0, 50, 60, 70] = 1
        res = compute_surface_dice(
            mask_pred, mask_gt, [1.0], include_background=True, spacing=(3, 2, 1), use_subvoxels=True
        )
        assert_allclose(res, 0.0, type_test=False)
        mask_gt[0, 0, 50, 60, 70] = 0
        mask_pred[0, 0, 50, 60, 72] = 1
        res = compute_surface_dice(
            mask_pred, mask_gt, [1.0], include_background=True, spacing=(3, 2, 1), use_subvoxels=True
        )
        assert_allclose(res, 0.0, type_test=False)
        mask_gt[0, 0, 50, 60, 70] = 1
        mask_pred[0, 0, 50, 60, 72] = 1
        res = compute_surface_dice(
            mask_pred, mask_gt, [1.0], include_background=True, spacing=(3, 2, 1), use_subvoxels=True
        )
        assert_allclose(res, 0.5, type_test=False)

        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask_gt, mask_pred = (torch.zeros(1, 1, 100, 100, 100, device=d), torch.zeros(1, 1, 100, 100, 100, device=d))
        mask_gt[0, 0, 0:50, :, :] = 1
        mask_pred[0, 0, 0:51, :, :] = 1
        res = compute_surface_dice(
            mask_pred, mask_gt, [1.0], include_background=True, spacing=(2, 1, 1), use_subvoxels=True
        )
        assert_allclose(res, 0.836145, type_test=False, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
