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

from monai.transforms import RandScaleCropd
from tests.croppers import CropTest
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TEST_SHAPES = [
    [{"keys": "img", "roi_scale": [1.0, 1.0, -1.0], "random_center": True}, (3, 3, 3, 4), (3, 3, 3, 4)],
    [
        # test `allow_missing_keys` with key "label"
        {"keys": ["label", "img"], "roi_scale": [1.0, 1.0, 1.0], "random_center": False, "allow_missing_keys": True},
        (3, 3, 3, 3),
        (3, 3, 3, 3),
    ],
]

TEST_VALUES = [
    [
        {"keys": "img", "roi_scale": [0.6, 0.6], "random_center": False},
        np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
    ]
]

TEST_RANDOM_SHAPES = [
    [
        {
            "keys": "img",
            "roi_scale": [0.75, 0.6, 0.5],
            "max_roi_scale": [1.0, -1.0, 0.6],
            "random_center": True,
            "random_size": True,
        },
        (1, 4, 5, 6),
        (1, 3, 4, 3),
    ],
    [
        {"keys": "img", "roi_scale": 0.6, "max_roi_scale": 0.8, "random_center": True, "random_size": True},
        (1, 4, 5, 6),
        (1, 3, 4, 4),
    ],
    [
        {"keys": "img", "roi_scale": 0.2, "max_roi_scale": 0.8, "random_center": True, "random_size": True},
        (1, 4, 5, 6),
        (1, 3, 2, 4),
    ],
]


class TestRandScaleCropd(CropTest):
    Cropper = RandScaleCropd

    @parameterized.expand(TEST_SHAPES)
    def test_shape(self, input_param, input_shape, expected_shape):
        self.crop_test(input_param, input_shape, expected_shape)

    @parameterized.expand(TEST_VALUES)
    def test_value(self, input_param, input_im):
        for im_type in TEST_NDARRAYS_ALL:
            with self.subTest(im_type=im_type):
                cropper = self.Cropper(**input_param)
                input_data = {"img": im_type(input_im)}
                result = cropper(input_data)["img"]
                roi = [(2 - i // 2, 2 + i - i // 2) for i in cropper.cropper._size]
                assert_allclose(result, input_im[:, roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]], type_test="tensor")

    @parameterized.expand(TEST_RANDOM_SHAPES)
    def test_random_shape(self, input_param, input_shape, expected_shape):
        for im_type in TEST_NDARRAYS_ALL:
            with self.subTest(im_type=im_type):
                cropper = self.Cropper(**input_param)
                cropper.set_random_state(seed=123)
                input_data = {"img": im_type(np.random.randint(0, 2, input_shape))}
                result = cropper(input_data)["img"]
                self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand(TEST_SHAPES)
    def test_pending_ops(self, input_param, input_shape, _):
        self.crop_test_pending_ops(input_param, input_shape)


if __name__ == "__main__":
    unittest.main()
