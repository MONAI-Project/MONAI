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

from monai.transforms import LabelToContourd
from tests.util import TEST_NDARRAYS, assert_allclose

expected_output_for_cube = [
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
]


def gen_fixed_cube(array_type):
    scale, core_start, core_end = 8, 1, 7
    cube = np.zeros((scale, scale, scale))
    cube[core_start:core_end, core_start:core_end, core_start:core_end] = torch.ones(
        core_end - core_start, core_end - core_start, core_end - core_start
    )
    cube = cube[None]

    batch_size, channels = 10, 6
    cube = np.tile(cube, (batch_size, channels, 1, 1, 1))
    return array_type(cube), array_type(expected_output_for_cube)


def gen_fixed_img(array_type):
    img = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    batch_size, channels = 10, 6
    img = np.tile(img, (batch_size, channels, 1, 1))
    img = array_type(img)
    expected_output_for_img = array_type(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    return img, expected_output_for_img


class TestContourd(unittest.TestCase):
    def test_contour(self):
        input_param = {"keys": "img", "kernel_type": "Laplace"}

        for p in TEST_NDARRAYS:
            # check 5-dim input data
            test_cube, expected_output = gen_fixed_cube(p)
            for cube in test_cube:
                test_result_cube = LabelToContourd(**input_param)({"img": cube})
                self.assertEqual(test_result_cube["img"].shape, cube.shape)

                test_result_np = test_result_cube["img"]
                channels = cube.shape[0]
                for channel in range(channels):
                    assert_allclose(test_result_np[channel, ...], expected_output, type_test="tensor")

            # check 4-dim input data
            test_img, expected_output = gen_fixed_img(p)
            for img in test_img:
                channels = img.shape[0]
                test_result_img = LabelToContourd(**input_param)({"img": img})
                self.assertEqual(test_result_img["img"].shape, img.shape)

                test_result_np = test_result_img["img"]
                for channel in range(channels):
                    assert_allclose(test_result_np[channel, ...], expected_output, type_test="tensor")

        # check invalid input data
        error_input = {"img": torch.rand(1, 2)}
        self.assertRaises(ValueError, LabelToContourd(**input_param), error_input)
        error_input = {"img": np.random.rand(1, 2)}
        self.assertRaises(ValueError, LabelToContourd(**input_param), error_input)
        error_input = {"img": torch.rand(1, 2, 3, 4, 5)}
        self.assertRaises(ValueError, LabelToContourd(**input_param), error_input)


if __name__ == "__main__":
    unittest.main()
