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

from monai.transforms import ShiftIntensity, StdShiftIntensity
from monai.utils import dtype_numpy_to_torch
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestStdShiftIntensity(NumpyImageTestCase2D):
    def test_value(self):
        for p in TEST_NDARRAYS:
            imt = p(self.imt)
            factor = np.random.rand()
            offset = np.std(self.imt) * factor
            shifter = ShiftIntensity(offset=offset)
            expected = shifter(imt)
            std_shifter = StdShiftIntensity(factor=factor)
            result = std_shifter(imt)
            assert_allclose(result, expected, atol=0, rtol=1e-5, type_test=False)

    def test_zerostd(self):
        for p in TEST_NDARRAYS:
            image = p(np.ones([2, 3, 3], dtype=np.float32))
            for nonzero in [True, False]:
                for channel_wise in [True, False]:
                    factor = np.random.rand()
                    std_shifter = StdShiftIntensity(factor=factor, nonzero=nonzero, channel_wise=channel_wise)
                    result = std_shifter(image)
                    assert_allclose(result, image, atol=0, rtol=1e-5, type_test=False)

    def test_nonzero(self):
        for p in TEST_NDARRAYS:
            image = p(np.asarray([[4.0, 0.0, 2.0], [0, 2, 4]]))  # std = 1
            factor = np.random.rand()
            std_shifter = StdShiftIntensity(factor=factor, nonzero=True)
            result = std_shifter(image)
            expected = p(np.asarray([[4 + factor, 0, 2 + factor], [0, 2 + factor, 4 + factor]], dtype=np.float32))
            assert_allclose(result, expected, atol=0, rtol=1e-5, type_test=False)

    def test_channel_wise(self):
        for p in TEST_NDARRAYS:
            image = p(np.stack((np.asarray([1.0, 2.0]), np.asarray([1.0, 1.0]))))  # std: 0.5, 0
            factor = np.random.rand()
            std_shifter = StdShiftIntensity(factor=factor, channel_wise=True)
            result = std_shifter(image)
            expected = p(
                np.stack((np.asarray([1 + 0.5 * factor, 2 + 0.5 * factor]), np.asarray([1, 1]))).astype(np.float32)
            )
            assert_allclose(result, expected, atol=0, rtol=1e-5, type_test=False)

    def test_dtype(self):
        trans_dtype = np.float32
        for dtype in [int, np.float32, np.float64]:
            image = np.random.rand(2, 2, 2).astype(dtype)
            factor = np.random.rand()
            std_shifter = StdShiftIntensity(factor=factor, dtype=trans_dtype)
            result = std_shifter(image)
            np.testing.assert_equal(result.dtype, dtype_numpy_to_torch(trans_dtype))


if __name__ == "__main__":
    unittest.main()
