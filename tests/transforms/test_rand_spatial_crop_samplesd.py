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
from monai.transforms import Compose, DivisiblePadd, RandSpatialCropSamplesd
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, assert_allclose

TEST_CASE_1 = [
    {"keys": ["img", "seg"], "num_samples": 4, "roi_size": [2, 2, 2], "random_center": True, "random_size": True},
    {"img": np.arange(81).reshape(3, 3, 3, 3), "seg": np.arange(81, 0, -1).reshape(3, 3, 3, 3)},
    [(3, 2, 2, 2), (3, 2, 3, 3), (3, 2, 3, 2), (3, 2, 3, 2)],
    {
        "img": np.array(
            [
                [[[1, 2], [4, 5], [7, 8]], [[10, 11], [13, 14], [16, 17]]],
                [[[28, 29], [31, 32], [34, 35]], [[37, 38], [40, 41], [43, 44]]],
                [[[55, 56], [58, 59], [61, 62]], [[64, 65], [67, 68], [70, 71]]],
            ]
        ),
        "seg": np.array(
            [
                [[[80, 79], [77, 76], [74, 73]], [[71, 70], [68, 67], [65, 64]]],
                [[[53, 52], [50, 49], [47, 46]], [[44, 43], [41, 40], [38, 37]]],
                [[[26, 25], [23, 22], [20, 19]], [[17, 16], [14, 13], [11, 10]]],
            ]
        ),
    },
]

TEST_CASE_2 = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASE_2.append(
        [
            {
                "keys": ["img", "seg"],
                "num_samples": 8,
                "roi_size": [2, 2, 3],
                "random_center": False,
                "random_size": True,
            },
            {"img": p(np.arange(81).reshape(3, 3, 3, 3)), "seg": p(np.arange(81, 0, -1).reshape(3, 3, 3, 3))},
            [
                (3, 2, 2, 3),
                (3, 2, 2, 3),
                (3, 3, 3, 3),
                (3, 2, 3, 3),
                (3, 3, 3, 3),
                (3, 2, 3, 3),
                (3, 2, 3, 3),
                (3, 3, 2, 3),
            ],
            {
                "img": p(
                    np.array(
                        [
                            [[[0, 1, 2], [3, 4, 5]], [[9, 10, 11], [12, 13, 14]], [[18, 19, 20], [21, 22, 23]]],
                            [[[27, 28, 29], [30, 31, 32]], [[36, 37, 38], [39, 40, 41]], [[45, 46, 47], [48, 49, 50]]],
                            [[[54, 55, 56], [57, 58, 59]], [[63, 64, 65], [66, 67, 68]], [[72, 73, 74], [75, 76, 77]]],
                        ]
                    )
                ),
                "seg": p(
                    np.array(
                        [
                            [[[81, 80, 79], [78, 77, 76]], [[72, 71, 70], [69, 68, 67]], [[63, 62, 61], [60, 59, 58]]],
                            [[[54, 53, 52], [51, 50, 49]], [[45, 44, 43], [42, 41, 40]], [[36, 35, 34], [33, 32, 31]]],
                            [[[27, 26, 25], [24, 23, 22]], [[18, 17, 16], [15, 14, 13]], [[9, 8, 7], [6, 5, 4]]],
                        ]
                    )
                ),
            },
        ]
    )


class TestRandSpatialCropSamplesd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, *TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape, expected_last):
        xform = RandSpatialCropSamplesd(**input_param)
        xform.set_random_state(1234)
        result = xform(input_data)
        _len = len(tuple(input_data.keys()))
        self.assertTupleEqual(tuple(result[0].keys())[:_len], tuple(input_data.keys()))
        for item, expected in zip(result, expected_shape):
            self.assertTupleEqual(item["img"].shape, expected)
            self.assertTupleEqual(item["seg"].shape, expected)
        for i, item in enumerate(result):
            self.assertEqual(item["img"].meta["patch_index"], i)
            self.assertEqual(item["seg"].meta["patch_index"], i)
        assert_allclose(item["img"], expected_last["img"], type_test=False)
        assert_allclose(item["seg"], expected_last["seg"], type_test=False)

    def test_deep_copy(self):
        data = {"img": np.ones((1, 10, 11, 12))}
        num_samples = 3
        sampler = RandSpatialCropSamplesd(
            keys=["img"], roi_size=(3, 3, 3), num_samples=num_samples, random_center=True, random_size=False
        )
        transform = Compose([DivisiblePadd(keys="img", k=5), sampler])
        samples = transform(data)
        self.assertEqual(len(samples), num_samples)
        for sample in samples:
            self.assertEqual(len(sample["img"].applied_operations), len(transform))

    @parameterized.expand([TEST_CASE_1, *TEST_CASE_2])
    def test_pending_ops(self, input_param, input_data, _expected_shape, _expected_last):
        xform = RandSpatialCropSamplesd(**input_param)
        # non-lazy
        xform.set_random_state(1234)
        expected = xform(input_data)
        self.assertIsInstance(expected[0]["img"], MetaTensor)

        # lazy
        xform.set_random_state(1234)
        xform.lazy = True
        pending_result = xform(input_data)
        for i, _pending_result in enumerate(pending_result):
            self.assertIsInstance(_pending_result["img"], MetaTensor)
            assert_allclose(_pending_result["img"].peek_pending_affine(), expected[i]["img"].affine)
            assert_allclose(_pending_result["img"].peek_pending_shape(), expected[i]["img"].shape[1:])
            # only support nearest
            overrides = {"mode": "nearest", "align_corners": False}
            result_img = apply_pending(_pending_result["img"], overrides=overrides)[0]
            result_seg = apply_pending(_pending_result["seg"], overrides=overrides)[0]
            # compare
            assert_allclose(result_img, expected[i]["img"], rtol=1e-5)
            assert_allclose(result_seg, expected[i]["seg"], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
