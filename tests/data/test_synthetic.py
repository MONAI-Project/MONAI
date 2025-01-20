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
from parameterized import parameterized

from monai.data import create_test_image_2d, create_test_image_3d
from monai.utils import set_determinism

TEST_CASES = [
    [2, {"width": 64, "height": 64, "rad_max": 10, "rad_min": 4}, 0.1479004, 0.739502, (64, 64), 5],
    [
        2,
        {"width": 28, "height": 32, "num_objs": 3, "rad_max": 5, "rad_min": 1, "noise_max": 0.2},
        0.1709315,
        0.4040179,
        (32, 28),
        5,
    ],
    [
        3,
        {"width": 64, "height": 64, "depth": 45, "num_seg_classes": 3, "channel_dim": -1, "rad_max": 10, "rad_min": 4},
        0.025132,
        0.0753961,
        (64, 64, 45, 1),
        3,
    ],
]


class TestDiceCELoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_create_test_image(self, dim, input_param, expected_img, expected_seg, expected_shape, expected_max_cls):
        set_determinism(seed=0)
        if dim == 2:
            img, seg = create_test_image_2d(**input_param)
        else:  # dim == 3
            img, seg = create_test_image_3d(**input_param)
        self.assertEqual(img.shape, expected_shape)
        self.assertEqual(seg.max(), expected_max_cls)
        np.testing.assert_allclose(img.mean(), expected_img, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(seg.mean(), expected_seg, atol=1e-7, rtol=1e-7)

    def test_ill_radius(self):
        with self.assertRaisesRegex(ValueError, ""):
            img, seg = create_test_image_2d(32, 32, rad_max=20)
        with self.assertRaisesRegex(ValueError, ""):
            img, seg = create_test_image_3d(32, 32, 32, rad_max=10, rad_min=11)
        with self.assertRaisesRegex(ValueError, ""):
            img, seg = create_test_image_2d(32, 32, rad_max=10, rad_min=0)


if __name__ == "__main__":
    unittest.main()
