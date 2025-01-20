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
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
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
    [MetaTensor(DATA_2D, affine=AFFINE_1), POINT_2D_WORLD, None, True, False, POINT_2D_IMAGE],  # use image affine
    [None, MetaTensor(POINT_2D_IMAGE, affine=AFFINE_1), None, False, False, POINT_2D_WORLD],  # use point affine
    [None, MetaTensor(POINT_2D_IMAGE, affine=AFFINE_1), AFFINE_1, False, False, POINT_2D_WORLD],  # use input affine
    [None, POINT_2D_WORLD, AFFINE_1, True, False, POINT_2D_IMAGE],  # use input affine
    [
        MetaTensor(DATA_2D, affine=AFFINE_1),
        POINT_2D_WORLD,
        None,
        True,
        True,
        POINT_2D_IMAGE_RAS,
    ],  # test affine_lps_to_ras
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, None, True, False, POINT_3D_IMAGE],
    ["affine", POINT_3D_WORLD, None, True, False, POINT_3D_IMAGE],  # use refer_data itself
    [
        MetaTensor(DATA_3D, affine=AFFINE_2),
        MetaTensor(POINT_3D_IMAGE, affine=AFFINE_2),
        None,
        False,
        False,
        POINT_3D_WORLD,
    ],
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, None, True, True, POINT_3D_IMAGE_RAS],
    [MetaTensor(DATA_3D, affine=AFFINE_2), POINT_3D_WORLD, None, True, True, POINT_3D_IMAGE_RAS],
]
TEST_CASES_SEQUENCE = [
    [
        (MetaTensor(DATA_2D, affine=AFFINE_1), MetaTensor(DATA_3D, affine=AFFINE_2)),
        [POINT_2D_WORLD, POINT_3D_WORLD],
        None,
        True,
        False,
        ["image_1", "image_2"],
        [POINT_2D_IMAGE, POINT_3D_IMAGE],
    ],  # use image affine
    [
        (MetaTensor(DATA_2D, affine=AFFINE_1), MetaTensor(DATA_3D, affine=AFFINE_2)),
        [POINT_2D_WORLD, POINT_3D_WORLD],
        None,
        True,
        True,
        ["image_1", "image_2"],
        [POINT_2D_IMAGE_RAS, POINT_3D_IMAGE_RAS],
    ],  # test affine_lps_to_ras
    [
        (None, None),
        [MetaTensor(POINT_2D_IMAGE, affine=AFFINE_1), MetaTensor(POINT_3D_IMAGE, affine=AFFINE_2)],
        None,
        False,
        False,
        None,
        [POINT_2D_WORLD, POINT_3D_WORLD],
    ],  # use point affine
    [
        (None, None),
        [POINT_2D_WORLD, POINT_2D_WORLD],
        AFFINE_1,
        True,
        False,
        None,
        [POINT_2D_IMAGE, POINT_2D_IMAGE],
    ],  # use input affine
    [
        (MetaTensor(DATA_2D, affine=AFFINE_1), MetaTensor(DATA_3D, affine=AFFINE_2)),
        [MetaTensor(POINT_2D_IMAGE, affine=AFFINE_1), MetaTensor(POINT_3D_IMAGE, affine=AFFINE_2)],
        None,
        False,
        False,
        ["image_1", "image_2"],
        [POINT_2D_WORLD, POINT_3D_WORLD],
    ],
]

TEST_CASES_WRONG = [
    [POINT_2D_WORLD, True, None, None],
    [POINT_2D_WORLD.unsqueeze(0), False, None, None],
    [POINT_3D_WORLD[..., 0:1], False, None, None],
    [POINT_3D_WORLD, False, torch.tensor([[[1, 0, 0, 10], [0, 1, 0, -4], [0, 0, 1, 0], [0, 0, 0, 1]]]), None],
    [POINT_3D_WORLD, False, None, "image"],
    [POINT_3D_WORLD, False, None, []],
]


class TestCoordinateTransform(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_transform_coordinates(self, image, points, affine, invert_affine, affine_lps_to_ras, expected_output):
        data = {
            "image": image,
            "point": points,
            "affine": torch.tensor([[1, 0, 0, 10], [0, 1, 0, -4], [0, 0, 1, 0], [0, 0, 0, 1]]),
        }
        refer_keys = "image" if (image is not None and image != "affine") else image
        transform = ApplyTransformToPointsd(
            keys="point",
            refer_keys=refer_keys,
            dtype=torch.int64,
            affine=affine,
            invert_affine=invert_affine,
            affine_lps_to_ras=affine_lps_to_ras,
        )
        output = transform(data)

        self.assertTrue(torch.allclose(output["point"], expected_output))
        invert_out = transform.inverse(output)
        self.assertTrue(torch.allclose(invert_out["point"], points))

    @parameterized.expand(TEST_CASES_SEQUENCE)
    def test_transform_coordinates_sequences(
        self, image, points, affine, invert_affine, affine_lps_to_ras, refer_keys, expected_output
    ):
        data = {"image_1": image[0], "image_2": image[1], "point_1": points[0], "point_2": points[1]}
        keys = ["point_1", "point_2"]
        transform = ApplyTransformToPointsd(
            keys=keys,
            refer_keys=refer_keys,
            dtype=torch.int64,
            affine=affine,
            invert_affine=invert_affine,
            affine_lps_to_ras=affine_lps_to_ras,
        )
        output = transform(data)

        self.assertTrue(torch.allclose(output["point_1"], expected_output[0]))
        self.assertTrue(torch.allclose(output["point_2"], expected_output[1]))
        invert_out = transform.inverse(output)
        self.assertTrue(torch.allclose(invert_out["point_1"], points[0]))

    @parameterized.expand(TEST_CASES_WRONG)
    def test_wrong_input(self, input, invert_affine, affine, refer_keys):
        if refer_keys == []:
            with self.assertRaises(ValueError):
                ApplyTransformToPointsd(
                    keys="point", dtype=torch.int64, invert_affine=invert_affine, affine=affine, refer_keys=refer_keys
                )
        else:
            transform = ApplyTransformToPointsd(
                keys="point", dtype=torch.int64, invert_affine=invert_affine, affine=affine, refer_keys=refer_keys
            )
            data = {"point": input}
            if refer_keys == "image":
                with self.assertRaises(KeyError):
                    transform(data)
            else:
                with self.assertRaises(ValueError):
                    transform(data)


if __name__ == "__main__":
    unittest.main()
