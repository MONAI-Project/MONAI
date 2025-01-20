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

from monai.config import USE_COMPILED
from monai.data.meta_tensor import MetaTensor
from monai.transforms import CropForeground
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TEST_COORDS, TESTS, TEST_LAZY_ERROR = [], [], []

for p in TEST_NDARRAYS_ALL:
    TEST_COORDS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
            True,
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[3]]]),
            False,
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
            True,
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]]),
            True,
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1], "allow_smaller": True},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            True,
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1], "allow_smaller": False},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            True,
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0, "k_divisible": 4},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[1, 2, 1, 0], [2, 3, 2, 0], [1, 2, 1, 0], [0, 0, 0, 0]]]),
            True,
        ]
    )

    TEST_LAZY_ERROR.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0, "k_divisible": 10},
            p([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p(np.zeros((1, 0, 0), dtype=np.int64)),
            True,
        ]
    )


class TestCropForeground(unittest.TestCase):
    @parameterized.expand(TEST_COORDS + TESTS)
    def test_value(self, arguments, image, expected_data, _):
        cropper = CropForeground(**arguments)
        result = cropper(image)
        assert_allclose(result, expected_data, type_test=False)
        self.assertIsInstance(result, MetaTensor)
        self.assertEqual(len(result.applied_operations), 1)
        inv = cropper.inverse(result)
        self.assertIsInstance(inv, MetaTensor)
        self.assertEqual(inv.applied_operations, [])
        self.assertTupleEqual(inv.shape, image.shape)

    @parameterized.expand(TEST_COORDS)
    def test_return_coords(self, arguments, image, _expected_data, _align_corners):
        arguments["return_coords"] = True
        _, start_coord, end_coord = CropForeground(**arguments)(image)
        arguments["return_coords"] = False
        np.testing.assert_allclose(start_coord, np.asarray([1, 1]))
        np.testing.assert_allclose(end_coord, np.asarray([4, 4]))

    @parameterized.expand(TEST_COORDS + TESTS)
    def test_pending_ops(self, input_param, image, _expected_data, align_corners):
        crop_fn = CropForeground(**input_param)
        # non-lazy
        expected = crop_fn(image)
        self.assertIsInstance(expected, MetaTensor)
        # lazy
        crop_fn.lazy = True
        pending_result = crop_fn(image)
        self.assertIsInstance(pending_result, MetaTensor)
        assert_allclose(pending_result.peek_pending_affine(), expected.affine)
        assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
        # only support nearest
        overrides = {"mode": "nearest", "align_corners": align_corners}
        result = apply_pending(pending_result, overrides=overrides)[0]
        # compare
        assert_allclose(result, expected, rtol=1e-5)

    @parameterized.expand(TEST_LAZY_ERROR)
    @unittest.skipIf(USE_COMPILED, "skip errors whe use compiled")
    def test_lazy_error(self, input_param, image, _expected_data, align_corners):
        with self.assertRaises(ValueError):
            crop_fn = CropForeground(**input_param)
            # lazy
            crop_fn.lazy = True
            pending_result = crop_fn(image)
            overrides = {"mode": "nearest", "align_corners": align_corners}
            return apply_pending(pending_result, overrides=overrides)[0]

    @parameterized.expand(TEST_COORDS + TESTS)
    def test_inverse_pending_ops(self, input_param, image, _expected_data, align_corners):
        crop_fn = CropForeground(**input_param)
        crop_fn.lazy = True
        pending_result = crop_fn(image)
        self.assertIsInstance(pending_result, MetaTensor)
        result = apply_pending(pending_result, overrides={"mode": "nearest", "align_corners": align_corners})[0]
        inverted = crop_fn.inverse(result)
        self.assertEqual(image.shape, inverted.shape)
        self.assertTrue((not inverted.applied_operations) and (not inverted.pending_operations))


if __name__ == "__main__":
    unittest.main()
