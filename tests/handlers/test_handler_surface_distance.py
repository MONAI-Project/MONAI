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
from ignite.engine import Engine

from monai.handlers import SurfaceDistance
from tests.util import assert_allclose


def create_spherical_seg_3d(
    radius: float = 20.0, centre: tuple[int, int, int] = (49, 49, 49), im_shape: tuple[int, int, int] = (99, 99, 99)
) -> np.ndarray:
    """
    Return a 3D image with a sphere inside. Voxel values will be
    1 inside the sphere, and 0 elsewhere.

    Args:
        radius: radius of sphere (in terms of number of voxels, can be partial)
        centre: location of sphere centre.
        im_shape: shape of image to create

    See also:
        :py:meth:`~create_test_image_3d`
    """
    # Create image
    image = np.zeros(im_shape, dtype=np.int32)
    spy, spx, spz = np.ogrid[
        -centre[0] : im_shape[0] - centre[0], -centre[1] : im_shape[1] - centre[1], -centre[2] : im_shape[2] - centre[2]
    ]
    circle = (spx * spx + spy * spy + spz * spz) <= radius * radius

    image[circle] = 1
    image[~circle] = 0
    return image


sampler_sphere = torch.Tensor(create_spherical_seg_3d(radius=20, centre=(20, 20, 20))).unsqueeze(0).unsqueeze(0)
# test input a list of channel-first tensor
sampler_sphere_gt = [torch.Tensor(create_spherical_seg_3d(radius=20, centre=(10, 20, 20))).unsqueeze(0)]
sampler_sphere_zeros = torch.zeros_like(sampler_sphere)

TEST_SAMPLE_1 = [sampler_sphere, sampler_sphere_gt]
TEST_SAMPLE_2 = [sampler_sphere_gt, sampler_sphere_gt]
TEST_SAMPLE_3 = [sampler_sphere_zeros, sampler_sphere_gt]
TEST_SAMPLE_4 = [sampler_sphere_zeros, sampler_sphere_zeros]


class TestHandlerSurfaceDistance(unittest.TestCase):
    # TODO test multi node Surface Distance

    def test_compute(self):
        sur_metric = SurfaceDistance(include_background=True)

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        sur_metric.attach(engine, "surface_distance")

        y_pred, y = TEST_SAMPLE_1
        sur_metric.update([y_pred, y])
        self.assertAlmostEqual(sur_metric.compute(), 4.17133, places=4)
        y_pred, y = TEST_SAMPLE_2
        sur_metric.update([y_pred, y])
        self.assertAlmostEqual(sur_metric.compute(), 2.08566, places=4)
        y_pred, y = TEST_SAMPLE_3
        sur_metric.update([y_pred, y])
        self.assertAlmostEqual(sur_metric.compute(), float("inf"))
        y_pred, y = TEST_SAMPLE_4
        sur_metric.update([y_pred, y])
        self.assertAlmostEqual(sur_metric.compute(), float("inf"))

    def test_shape_mismatch(self):
        sur_metric = SurfaceDistance(include_background=True)
        with self.assertRaises((AssertionError, ValueError)):
            y_pred = TEST_SAMPLE_1[0]
            y = torch.ones((1, 1, 10, 10, 10))
            sur_metric.update([y_pred, y])

    def test_reduction(self):
        sur_metric = SurfaceDistance(include_background=True, reduction="mean_channel")

        def _val_func(engine, batch):
            pass

        engine = Engine(_val_func)
        sur_metric.attach(engine, "surface_distance")

        y_pred, y = TEST_SAMPLE_1
        sur_metric.update([y_pred, y])
        y_pred, y = TEST_SAMPLE_2
        sur_metric.update([y_pred, y])
        assert_allclose(
            sur_metric.compute().float(), torch.tensor([4.1713, 0.0000]), atol=1e-4, rtol=1e-4, type_test=False
        )


if __name__ == "__main__":
    unittest.main()
