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

import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms.utility.array import ApplyTransformToPoints
from monai.utils import set_determinism

set_determinism(seed=0)

DATA_2D = torch.rand(1, 64, 64)
DATA_3D = torch.rand(1, 64, 64, 64)
POINT_2D_WORLD = torch.tensor([[[2, 2], [2, 4], [4, 6]]])
POINT_2D_IMAGE = torch.tensor([[[1, 1], [1, 2], [2, 3]]])
POINT_2D_IMAGE_RAS = torch.tensor([[[-1, -1], [-1, -2], [-2, -3]]])
POINT_3D_WORLD = torch.tensor([[[2, 4, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]])
POINT_3D_IMAGE = torch.tensor([[[-8, 8, 6], [-2, 14, 12]], [[4, 20, 18], [10, 26, 24]]])
POINT_3D_IMAGE_RAS = torch.tensor([[[-12, 0, 6], [-18, -6, 12]], [[-24, -12, 18], [-30, -18, 24]]])
AFFINE_1 = torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
AFFINE_2 = torch.tensor([[1, 0, 0, 10], [0, 1, 0, -4], [0, 0, 1, 0], [0, 0, 0, 1]])

TEST_CASES = [
    [MetaTensor(DATA_2D, affine=AFFINE_1), POINT_2D_WORLD, None, True, False, POINT_2D_IMAGE],
    [None, MetaTensor(POINT_2D_IMAGE, affine=AFFINE_1), None, False, False, POINT_2D_WORLD],
    [None, MetaTensor(POINT_2D_IMAGE, affine=AFFINE_1), AFFINE_1, False, False, POINT_2D_WORLD],
    [MetaTensor(DATA_2D, affine=AFFINE_1), POINT_2D_WORLD, None, True, True, POINT_2D_IMAGE_RAS],
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, None, True, False, POINT_3D_IMAGE],
    [
        MetaTensor(DATA_3D, affine=AFFINE_2),
        MetaTensor(POINT_3D_IMAGE, affine=AFFINE_2),
        None,
        False,
        False,
        POINT_3D_WORLD,
    ],
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, None, True, True, POINT_3D_IMAGE_RAS],
]

TEST_CASES_WRONG = [
    [POINT_2D_WORLD, True, None],
    [POINT_2D_WORLD.unsqueeze(0), False, None],
    [POINT_3D_WORLD[..., 0:1], False, None],
    [POINT_3D_WORLD, False, torch.tensor([[[1, 0, 0, 10], [0, 1, 0, -4], [0, 0, 1, 0], [0, 0, 0, 1]]])],
]


class TestCoordinateTransform(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transform_coordinates(self, image, points, affine, invert_affine, affine_lps_to_ras, expected_output):
        transform = ApplyTransformToPoints(
            dtype=torch.int64, affine=affine, invert_affine=invert_affine, affine_lps_to_ras=affine_lps_to_ras
        )
        affine = image.affine if image is not None else None
        output = transform(points, affine)
        self.assertTrue(torch.allclose(output, expected_output))
        invert_out = transform.inverse(output)
        self.assertTrue(torch.allclose(invert_out, points))

    @parameterized.expand(TEST_CASES_WRONG)
    def test_wrong_input(self, input, invert_affine, affine):
        transform = ApplyTransformToPoints(dtype=torch.int64, invert_affine=invert_affine)
        with self.assertRaises(ValueError):
            transform(input, affine)


if __name__ == "__main__":
    unittest.main()
