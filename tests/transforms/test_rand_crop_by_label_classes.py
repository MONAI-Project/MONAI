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
from monai.transforms import ClassesToIndices, RandCropByLabelClasses
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TESTS_INDICES, TESTS_SHAPE = [], []
for p in TEST_NDARRAYS_ALL:
    # One-Hot label
    TESTS_INDICES.append(
        [
            {
                "label": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "num_classes": None,
                "spatial_size": [2, 2, -1],
                "ratios": [1, 1, 1],
                "num_samples": 2,
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image_threshold": 0,
            },
            {"img": p(np.random.randint(0, 2, size=[3, 3, 3, 3]))},
            list,
            (3, 2, 2, 3),
        ]
    )

    TESTS_INDICES.append(
        [
            # Argmax label
            {
                "label": p(np.random.randint(0, 2, size=[1, 3, 3, 3])),
                "num_classes": 2,
                "spatial_size": [2, 2, 2],
                "ratios": [1, 1],
                "num_samples": 2,
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image_threshold": 0,
            },
            {"img": p(np.random.randint(0, 2, size=[3, 3, 3, 3]))},
            list,
            (3, 2, 2, 2),
        ]
    )

    TESTS_SHAPE.append(
        [
            # provide label at runtime
            {
                "label": None,
                "num_classes": 2,
                "spatial_size": [2, 2, 2],
                "ratios": [1, 1],
                "num_samples": 2,
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image_threshold": 0,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 2, size=[1, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
            },
            list,
            (3, 2, 2, 2),
        ]
    )
    TESTS_SHAPE.append(
        [
            # provide label at runtime
            {
                "label": None,
                "num_classes": 2,
                "spatial_size": [4, 4, 2],
                "ratios": [1, 1],
                "num_samples": 2,
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image_threshold": 0,
                "allow_smaller": True,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 2, size=[1, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
            },
            list,
            (3, 3, 3, 2),
        ]
    )
    TESTS_SHAPE.append(
        [
            # provide label at runtime
            {
                "label": None,
                "num_classes": 2,
                "spatial_size": [4, 4, 4],
                "ratios": (1, 1),  # test no assignment
                "num_samples": 2,
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image_threshold": 0,
                "allow_smaller": True,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 1, size=[1, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
            },
            list,
            (3, 3, 3, 3),
        ]
    )


class TestRandCropByLabelClasses(unittest.TestCase):
    @parameterized.expand(TESTS_INDICES + TESTS_SHAPE)
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = RandCropByLabelClasses(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)

    @parameterized.expand(TESTS_INDICES)
    def test_indices(self, input_param, input_data, expected_type, expected_shape):
        input_param["indices"] = ClassesToIndices(num_classes=input_param["num_classes"])(input_param["label"])
        result = RandCropByLabelClasses(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)
        # test set indices at runtime
        input_data["indices"] = input_param["indices"]
        result = RandCropByLabelClasses(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)

    @parameterized.expand(TESTS_INDICES + TESTS_SHAPE)
    def test_pending_ops(self, input_param, input_data, _expected_type, _expected_shape):
        cropper = RandCropByLabelClasses(**input_param)
        # non-lazy
        cropper.set_random_state(0)
        expected = cropper(**input_data)
        self.assertIsInstance(expected[0], MetaTensor)
        # lazy
        cropper.set_random_state(0)
        cropper.lazy = True
        pending_result = cropper(**input_data)
        for i, _pending_result in enumerate(pending_result):
            self.assertIsInstance(_pending_result, MetaTensor)
            assert_allclose(_pending_result.peek_pending_affine(), expected[i].affine)
            assert_allclose(_pending_result.peek_pending_shape(), expected[i].shape[1:])
            # only support nearest
            result = apply_pending(_pending_result, overrides={"mode": "nearest", "align_corners": False})[0]
            # compare
            assert_allclose(result, expected[i], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
