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

from monai.transforms import NormalizeIntensityd
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        TESTS.append(
            [
                {"keys": ["img"], "nonzero": True},
                {"img": p(np.array([0.0, 3.0, 0.0, 4.0]))},
                p(np.array([0.0, -1.0, 0.0, 1.0])),
            ]
        )
        TESTS.append(
            [
                {
                    "keys": ["img"],
                    "subtrahend": q(np.array([3.5, 3.5, 3.5, 3.5])),
                    "divisor": q(np.array([0.5, 0.5, 0.5, 0.5])),
                    "nonzero": True,
                },
                {"img": p(np.array([0.0, 3.0, 0.0, 4.0]))},
                p(np.array([0.0, -1.0, 0.0, 1.0])),
            ]
        )
        TESTS.append(
            [
                {"keys": ["img"], "nonzero": True},
                {"img": p(np.array([0.0, 0.0, 0.0, 0.0]))},
                p(np.array([0.0, 0.0, 0.0, 0.0])),
            ]
        )


class TestNormalizeIntensityd(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_image_normalize_intensityd(self, im_type):
        key = "img"
        im = im_type(self.imt)
        normalizer = NormalizeIntensityd(keys=[key])
        normalized = normalizer({key: im})[key]
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        assert_allclose(normalized, im_type(expected), rtol=1e-3, type_test="tensor")

    @parameterized.expand(TESTS)
    def test_nonzero(self, input_param, input_data, expected_data):
        key = "img"
        normalizer = NormalizeIntensityd(**input_param)
        normalized = normalizer(input_data)[key]
        assert_allclose(normalized, expected_data, type_test="tensor")

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, im_type):
        key = "img"
        normalizer = NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True)
        input_data = {key: im_type(np.array([[0.0, 3.0, 0.0, 4.0], [0.0, 4.0, 0.0, 5.0]]))}
        normalized = normalizer(input_data)[key]
        expected = np.array([[0.0, -1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0]])
        assert_allclose(normalized, im_type(expected), type_test="tensor")


if __name__ == "__main__":
    unittest.main()
