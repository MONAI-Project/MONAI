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
from itertools import product

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import HausdorffDistanceMetric

_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")


def create_spherical_seg_3d(
    radius: float = 20.0,
    centre: tuple[int, int, int] = (49, 49, 49),
    im_shape: tuple[int, int, int] = (99, 99, 99),
    im_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Return a 3D image with a sphere inside. Voxel values will be
    1 inside the sphere, and 0 elsewhere.

    Args:
        radius: radius of sphere (in terms of number of voxels, can be partial)
        centre: location of sphere centre.
        im_shape: shape of image to create.
        im_spacing: spacing of image to create.

    See also:
        :py:meth:`~create_test_image_3d`
    """
    # Create image
    image = np.zeros(im_shape, dtype=np.int32)
    spy, spx, spz = np.ogrid[: im_shape[0], : im_shape[1], : im_shape[2]]
    spy = spy.astype(float) * im_spacing[0]
    spx = spx.astype(float) * im_spacing[1]
    spz = spz.astype(float) * im_spacing[2]

    spy -= centre[0]
    spx -= centre[1]
    spz -= centre[2]

    circle = (spx * spx + spy * spy + spz * spz) <= radius * radius

    image[circle] = 1
    image[~circle] = 0
    return image


test_spacing = (0.85, 1.2, 0.9)
TEST_CASES = [
    [[create_spherical_seg_3d(), create_spherical_seg_3d(), None, 1], [0, 0, 0, 0, 0, 0]],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20)),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19)),
            None,
        ],
        [1.7320508075688772, 1.7320508075688772, 1, 1, 3, 3],
    ],
    [
        [
            create_spherical_seg_3d(radius=33, centre=(19, 33, 22)),
            create_spherical_seg_3d(radius=33, centre=(20, 33, 22)),
            None,
        ],
        [1, 1, 1, 1, 1, 1],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            None,
        ],
        [20.09975124224178, 20.223748416156685, 15, 20, 24, 35],
    ],
    [
        [
            # pred does not have foreground (but gt has), the metric should be inf
            np.zeros([99, 99, 99]),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            None,
        ],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    ],
    [
        [
            # gt does not have foreground (but pred has), the metric should be inf
            create_spherical_seg_3d(),
            np.zeros([99, 99, 99]),
            None,
        ],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            None,
            95,
        ],
        [19.924858845171276, 20.09975124224178, 14, 18, 22, 33],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20), im_spacing=test_spacing),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19), im_spacing=test_spacing),
            test_spacing,
        ],
        [2.0808651447296143, 2.2671568, 2, 2, 3, 4],
    ],
    [
        [
            create_spherical_seg_3d(radius=15, centre=(20, 33, 22), im_spacing=test_spacing),
            create_spherical_seg_3d(radius=30, centre=(20, 33, 22), im_spacing=test_spacing),
            test_spacing,
        ],
        [15.439640998840332, 15.62594, 11, 17, 20, 28],
    ],
]

TEST_CASES_NANS = [
    [
        [
            # both pred and gt do not have foreground, spacing is None, metric and not_nans should be 0
            np.zeros([99, 99, 99]),
            np.zeros([99, 99, 99]),
            None,
        ]
    ],
    [
        [
            # both pred and gt do not have foreground, metric and not_nans should be 0
            np.zeros([99, 99, 99]),
            np.zeros([99, 99, 99]),
            test_spacing,
        ]
    ],
]

TEST_CASES_EXPANDED = []
for test_case in TEST_CASES:
    test_output: list[float | int]
    test_input, test_output = test_case  # type: ignore
    for _device in _devices:
        for i, (metric, directed) in enumerate(product(["euclidean", "chessboard", "taxicab"], [True, False])):
            TEST_CASES_EXPANDED.append((_device, metric, directed, test_input, test_output[i]))


def _describe_test_case(test_func, test_number, params):
    _device, metric, directed, test_input, test_output = params.args
    return f"device: {_device} metric: {metric} directed:{directed} expected: {test_output}"


class TestHausdorffDistance(unittest.TestCase):

    @parameterized.expand(TEST_CASES_EXPANDED, doc_func=_describe_test_case)
    def test_value(self, device, metric, directed, input_data, expected_value):
        percentile = None
        if len(input_data) == 4:
            [seg_1, seg_2, spacing, percentile] = input_data
        else:
            [seg_1, seg_2, spacing] = input_data

        seg_1 = torch.tensor(seg_1, device=device)
        seg_2 = torch.tensor(seg_2, device=device)
        hd_metric = HausdorffDistanceMetric(
            include_background=False, distance_metric=metric, percentile=percentile, directed=directed
        )
        # shape of seg_1, seg_2 are: HWD, converts to BNHWD
        batch, n_class = 2, 3
        batch_seg_1 = seg_1.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
        batch_seg_2 = seg_2.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
        hd_metric(batch_seg_1, batch_seg_2, spacing=spacing)
        result: torch.Tensor = hd_metric.aggregate(reduction="mean")  # type: ignore
        np.testing.assert_allclose(expected_value, result.cpu(), rtol=1e-6)
        np.testing.assert_equal(result.device, seg_1.device)

    @parameterized.expand(TEST_CASES_NANS)
    def test_nans(self, input_data):
        [seg_1, seg_2, spacing] = input_data
        seg_1 = torch.tensor(seg_1)
        seg_2 = torch.tensor(seg_2)
        hd_metric = HausdorffDistanceMetric(include_background=False, get_not_nans=True)
        batch_seg_1 = seg_1.unsqueeze(0).unsqueeze(0)
        batch_seg_2 = seg_2.unsqueeze(0).unsqueeze(0)
        hd_metric(batch_seg_1, batch_seg_2, spacing=spacing)
        result, not_nans = hd_metric.aggregate()
        np.testing.assert_allclose(0, result, rtol=1e-7)
        np.testing.assert_allclose(0, not_nans, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
