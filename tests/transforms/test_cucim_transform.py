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

from monai.transforms import CuCIM
from monai.utils import optional_import, set_determinism
from tests.util import HAS_CUPY, skip_if_no_cuda

_, has_cut = optional_import("cucim.core.operations.expose.transform")
cp, _ = optional_import("cupy")

set_determinism(seed=0)

TEST_CASE_COLOR_JITTER_1 = [
    {"name": "color_jitter", "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
    np.array([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]], dtype=np.float32),
    np.array([[[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]], dtype=np.float32),
]

TEST_CASE_COLOR_JITTER_2 = [
    {"name": "color_jitter", "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
    np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]], dtype=np.uint8),
    np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]], dtype=np.uint8),
]

TEST_CASE_FLIP_1 = [
    {"name": "image_flip", "spatial_axis": -1},
    np.array([[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
    np.array([[[1.0, 0.0], [3.0, 2.0]], [[1.0, 0.0], [3.0, 2.0]], [[1.0, 0.0], [3.0, 2.0]]], dtype=np.float32),
]

TEST_CASE_ROTATE_1 = [
    {"name": "image_rotate_90", "k": 1, "spatial_axis": (-2, -1)},
    np.array([[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
    np.array([[[1.0, 3.0], [0.0, 2.0]], [[1.0, 3.0], [0.0, 2.0]], [[1.0, 3.0], [0.0, 2.0]]], dtype=np.float32),
]

TEST_CASE_SCALE_INTENSITY_1 = [
    {"name": "scale_intensity_range", "a_min": 0.0, "a_max": 4.0, "b_min": 0.0, "b_max": 1.0, "clip": False},
    np.array([[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
    np.array([[[0.0, 0.25], [0.5, 0.75]], [[0.0, 0.25], [0.5, 0.75]], [[0.0, 0.25], [0.5, 0.75]]], dtype=np.float32),
]

TEST_CASE_ZOOM_1 = [
    {"name": "zoom", "zoom_factor": (0.5, 0.5)},
    np.mgrid[:3, 1:4].astype(dtype=np.float32),
    np.concatenate([np.ones((1, 3, 3), dtype=np.float32) * 1.0, np.ones((1, 3, 3), dtype=np.float32) * 2.0]),
]


@skip_if_no_cuda
@unittest.skipUnless(HAS_CUPY, "CuPy is required.")
@unittest.skipUnless(has_cut, "cuCIM transforms are required.")
class TestCuCIM(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_COLOR_JITTER_1,
            TEST_CASE_COLOR_JITTER_2,
            TEST_CASE_FLIP_1,
            TEST_CASE_ROTATE_1,
            TEST_CASE_SCALE_INTENSITY_1,
            TEST_CASE_ZOOM_1,
        ]
    )
    def test_tramsforms_numpy_single(self, params, input, expected):
        output = CuCIM(**params)(input)
        self.assertEqual(output.dtype, expected.dtype)
        self.assertIsInstance(output, np.ndarray)
        cp.testing.assert_allclose(output, expected)

    @parameterized.expand(
        [
            TEST_CASE_COLOR_JITTER_1,
            TEST_CASE_COLOR_JITTER_2,
            TEST_CASE_FLIP_1,
            TEST_CASE_ROTATE_1,
            TEST_CASE_SCALE_INTENSITY_1,
            TEST_CASE_ZOOM_1,
        ]
    )
    def test_tramsforms_numpy_batch(self, params, input, expected):
        input = input[cp.newaxis, ...]
        expected = expected[cp.newaxis, ...]
        output = CuCIM(**params)(input)
        self.assertEqual(output.dtype, expected.dtype)
        self.assertIsInstance(output, np.ndarray)
        cp.testing.assert_allclose(output, expected)

    @parameterized.expand(
        [
            TEST_CASE_COLOR_JITTER_1,
            TEST_CASE_COLOR_JITTER_2,
            TEST_CASE_FLIP_1,
            TEST_CASE_ROTATE_1,
            TEST_CASE_SCALE_INTENSITY_1,
            TEST_CASE_ZOOM_1,
        ]
    )
    def test_tramsforms_cupy_single(self, params, input, expected):
        input = cp.asarray(input)
        expected = cp.asarray(expected)
        output = CuCIM(**params)(input)
        self.assertEqual(output.dtype, expected.dtype)
        self.assertIsInstance(output, cp.ndarray)
        cp.testing.assert_allclose(output, expected)

    @parameterized.expand(
        [
            TEST_CASE_COLOR_JITTER_1,
            TEST_CASE_COLOR_JITTER_2,
            TEST_CASE_FLIP_1,
            TEST_CASE_ROTATE_1,
            TEST_CASE_SCALE_INTENSITY_1,
            TEST_CASE_ZOOM_1,
        ]
    )
    def test_tramsforms_cupy_batch(self, params, input, expected):
        input = cp.asarray(input)[cp.newaxis, ...]
        expected = cp.asarray(expected)[cp.newaxis, ...]
        output = CuCIM(**params)(input)
        self.assertEqual(output.dtype, expected.dtype)
        self.assertIsInstance(output, cp.ndarray)
        cp.testing.assert_allclose(output, expected)


if __name__ == "__main__":
    unittest.main()
