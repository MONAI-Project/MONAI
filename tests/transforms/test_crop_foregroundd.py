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

from monai.data.meta_tensor import MetaTensor
from monai.transforms import CropForegroundd
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TEST_POSITION, TESTS = [], []
for p in TEST_NDARRAYS_ALL:
    TEST_POSITION.append(
        [
            {
                "keys": ["img", "label"],
                "source_key": "label",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            {
                "img": p(
                    np.array([[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]])
                ),
                "label": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]])
                ),
            },
            p(np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])),
            True,
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[3]]])),
            False,
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])),
            True,
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]])),
            True,
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": True,
            },
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])),
            True,
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": False,
            },
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 2, 1, 0],
                            [0, 2, 3, 2, 0],
                            [0, 1, 2, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ]
                )
            ),
            True,
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": 0,
                "margin": 0,
                "k_divisible": [4, 6],
                "mode": "constant",
            },
            {
                "img": p(
                    np.array(
                        [[[0, 2, 1, 2, 0], [1, 1, 2, 1, 1], [2, 2, 3, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0]]],
                        dtype=np.float32,
                    )
                )
            },
            p(np.array([[[0, 2, 1, 2, 0, 0], [1, 1, 2, 1, 1, 0], [2, 2, 3, 2, 2, 0], [1, 1, 2, 1, 1, 0]]])),
            False,
        ]
    )


class TestCropForegroundd(unittest.TestCase):
    @parameterized.expand(TEST_POSITION + TESTS)
    def test_value(self, arguments, input_data, expected_data, _):
        cropper = CropForegroundd(**arguments)
        result = cropper(input_data)
        assert_allclose(result["img"], expected_data, type_test="tensor")
        if "label" in input_data and "img" in input_data:
            self.assertTupleEqual(result["img"].shape, result["label"].shape)
        inv = cropper.inverse(result)
        self.assertTupleEqual(inv["img"].shape, input_data["img"].shape)
        if "label" in input_data:
            self.assertTupleEqual(inv["label"].shape, input_data["label"].shape)

    @parameterized.expand(TEST_POSITION)
    def test_foreground_position(self, arguments, input_data, _expected_data, _align_corners):
        result = CropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["foreground_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["foreground_end_coord"], np.array([4, 4]))

        arguments["start_coord_key"] = "test_start_coord"
        arguments["end_coord_key"] = "test_end_coord"
        result = CropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["test_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["test_end_coord"], np.array([4, 4]))

    @parameterized.expand(TEST_POSITION + TESTS)
    def test_pending_ops(self, input_param, image, _expected_data, align_corners):
        crop_fn = CropForegroundd(**input_param)
        # non-lazy
        expected = crop_fn(image)["img"]
        self.assertIsInstance(expected, MetaTensor)
        # lazy
        crop_fn.lazy = True
        pending_result = crop_fn(image)["img"]
        self.assertIsInstance(pending_result, MetaTensor)
        assert_allclose(pending_result.peek_pending_affine(), expected.affine)
        assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
        # only support nearest
        overrides = {"mode": "nearest", "align_corners": align_corners}
        result = apply_pending(pending_result, overrides=overrides)[0]
        # compare
        assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
