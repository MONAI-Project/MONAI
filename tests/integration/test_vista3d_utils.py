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
from unittest.case import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms.utils import convert_points_to_disc, keep_merge_components_with_points, sample_points_from_label
from monai.utils import min_version
from monai.utils.module import optional_import
from tests.util import skip_if_no_cuda, skip_if_quick

cp, has_cp = optional_import("cupy")
cucim_skimage, has_cucim = optional_import("cucim.skimage")
measure, has_measure = optional_import("skimage.measure", "0.14.2", min_version)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TESTS_SAMPLE_POINTS_FROM_LABEL = []
for use_center in [True, False]:
    labels = torch.zeros(1, 1, 32, 32, 32)
    labels[0, 0, 5:10, 5:10, 5:10] = 1
    labels[0, 0, 10:15, 10:15, 10:15] = 3
    labels[0, 0, 20:25, 20:25, 20:25] = 5
    TESTS_SAMPLE_POINTS_FROM_LABEL.append(
        [{"labels": labels, "label_set": (1, 3, 5), "use_center": use_center}, (3, 1, 3), (3, 1)]
    )

TEST_CONVERT_POINTS_TO_DISC = []
for radius in [1, 2]:
    for disc in [True, False]:
        image_size = (32, 32, 32)
        point = torch.randn(3, 1, 3)
        point_label = torch.randint(0, 4, (3, 1))
        expected_shape = (point.shape[0], 2, *image_size)
        TEST_CONVERT_POINTS_TO_DISC.append(
            [
                {"image_size": image_size, "point": point, "point_label": point_label, "radius": radius, "disc": disc},
                expected_shape,
            ]
        )
        image_size = (16, 32, 64)
        point = torch.tensor([[[8, 16, 42], [2, 8, 21]]])
        point_label = torch.tensor([[1, 0]])
        expected_shape = (point.shape[0], 2, *image_size)
        TEST_CONVERT_POINTS_TO_DISC.append(
            [
                {"image_size": image_size, "point": point, "point_label": point_label, "radius": radius, "disc": disc},
                expected_shape,
            ]
        )

TEST_CONVERT_POINTS_TO_DISC_VALUE = []
image_size = (16, 32, 64)
point = torch.tensor([[[8, 16, 42], [2, 8, 21]]])
point_label = torch.tensor([[1, 0]])
expected_shape = (point.shape[0], 2, *image_size)
for radius in [5, 10]:
    for disc in [True, False]:
        TEST_CONVERT_POINTS_TO_DISC_VALUE.append(
            [
                {"image_size": image_size, "point": point, "point_label": point_label, "radius": radius, "disc": disc},
                [point, point_label],
            ]
        )


TEST_LCC_MASK_POINT_TORCH = []
for bs in [1, 2]:
    for num_points in [1, 3]:
        shape = (bs, 1, 128, 32, 32)
        TEST_LCC_MASK_POINT_TORCH.append(
            [
                {
                    "img_pos": torch.randint(0, 2, shape, dtype=torch.bool),
                    "img_neg": torch.randint(0, 2, shape, dtype=torch.bool),
                    "point_coords": torch.randint(0, 10, (bs, num_points, 3)),
                    "point_labels": torch.randint(0, 4, (bs, num_points)),
                },
                shape,
            ]
        )

TEST_LCC_MASK_POINT_NP = []
for bs in [1, 2]:
    for num_points in [1, 3]:
        shape = (bs, 1, 32, 32, 64)
        TEST_LCC_MASK_POINT_NP.append(
            [
                {
                    "img_pos": np.random.randint(0, 2, shape, dtype=bool),
                    "img_neg": np.random.randint(0, 2, shape, dtype=bool),
                    "point_coords": np.random.randint(0, 5, (bs, num_points, 3)),
                    "point_labels": np.random.randint(0, 4, (bs, num_points)),
                },
                shape,
            ]
        )


@skipUnless(has_measure or cucim_skimage, "skimage or cucim.skimage required")
class TestSamplePointsFromLabel(unittest.TestCase):
    @parameterized.expand(TESTS_SAMPLE_POINTS_FROM_LABEL)
    def test_shape(self, input_data, expected_point_shape, expected_point_label_shape):
        point, point_label = sample_points_from_label(**input_data)
        self.assertEqual(point.shape, expected_point_shape)
        self.assertEqual(point_label.shape, expected_point_label_shape)


class TestConvertPointsToDisc(unittest.TestCase):
    @parameterized.expand(TEST_CONVERT_POINTS_TO_DISC)
    def test_shape(self, input_data, expected_shape):
        result = convert_points_to_disc(**input_data)
        self.assertEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_CONVERT_POINTS_TO_DISC_VALUE)
    def test_value(self, input_data, points):
        result = convert_points_to_disc(**input_data)
        point, point_label = points
        for i in range(point.shape[0]):
            for j in range(point.shape[1]):
                self.assertEqual(result[i, point_label[i, j], point[i, j][0], point[i, j][1], point[i, j][2]], True)


@skipUnless(has_measure or cucim_skimage, "skimage or cucim.skimage required")
class TestKeepMergeComponentsWithPoints(unittest.TestCase):
    @skip_if_quick
    @skip_if_no_cuda
    @skipUnless(has_cp and cucim_skimage, "cupy and cucim.skimage required")
    @parameterized.expand(TEST_LCC_MASK_POINT_TORCH)
    def test_cp_shape(self, input_data, shape):
        for key in input_data:
            input_data[key] = input_data[key].to(device)
        mask = keep_merge_components_with_points(**input_data)
        self.assertEqual(mask.shape, shape)

    @skipUnless(has_measure, "skimage required")
    @parameterized.expand(TEST_LCC_MASK_POINT_NP)
    def test_np_shape(self, input_data, shape):
        mask = keep_merge_components_with_points(**input_data)
        self.assertEqual(mask.shape, shape)


if __name__ == "__main__":
    unittest.main()
