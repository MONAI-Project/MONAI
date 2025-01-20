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

from monai.transforms import ScaleIntensity
from monai.transforms.utils import rescale_array
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestScaleIntensity(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_range_scale(self, p):
        scaler = ScaleIntensity(minv=1.0, maxv=2.0)
        im = p(self.imt)
        result = scaler(im)
        mina = self.imt.min()
        maxa = self.imt.max()
        norm = (self.imt - mina) / (maxa - mina)
        expected = p((norm * (2.0 - 1.0)) + 1.0)
        assert_allclose(result, expected, type_test="tensor", rtol=1e-7, atol=0)

    def test_factor_scale(self):
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensity(minv=None, maxv=None, factor=0.1)
            result = scaler(p(self.imt))
            expected = p((self.imt * (1 + 0.1)).astype(np.float32))
            assert_allclose(result, p(expected), type_test="tensor", rtol=1e-7, atol=0)

    def test_max_none(self):
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensity(minv=0.0, maxv=None, factor=0.1)
            result = scaler(p(self.imt))
            expected = rescale_array(p(self.imt), minv=0.0, maxv=None)
            assert_allclose(result, expected, type_test="tensor", rtol=1e-3, atol=1e-3)

    def test_int(self):
        """integers should be handled by converting them to floats first."""
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensity(minv=1.0, maxv=2.0)
            result = scaler(p(self.imt.astype(int)))
            _imt = self.imt.astype(int).astype(np.float32)
            mina = _imt.min()
            maxa = _imt.max()
            norm = (_imt - mina) / (maxa - mina)
            expected = p((norm * (2.0 - 1.0)) + 1.0)
            assert_allclose(result, expected, type_test="tensor", rtol=1e-7, atol=0)

    def test_channel_wise(self):
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensity(minv=1.0, maxv=2.0, channel_wise=True)
            data = p(np.tile(self.imt, (3, 1, 1, 1)))
            result = scaler(data)
            mina = self.imt.min()
            maxa = self.imt.max()
            for i, c in enumerate(data):
                norm = (c - mina) / (maxa - mina)
                expected = p((norm * (2.0 - 1.0)) + 1.0)
                assert_allclose(result[i], expected, type_test="tensor", rtol=1e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
