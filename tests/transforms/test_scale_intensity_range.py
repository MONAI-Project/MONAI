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

from monai.transforms import ScaleIntensityRange
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class IntensityScaleIntensityRange(NumpyImageTestCase2D):
    def test_image_scale_intensity_range(self):
        scaler = ScaleIntensityRange(a_min=20, a_max=108, b_min=50, b_max=80, dtype=np.uint8)
        for p in TEST_NDARRAYS:
            scaled = scaler(p(self.imt))
            self.assertTrue(scaled.dtype, np.uint8)
            expected = (((self.imt - 20) / 88) * 30 + 50).astype(np.uint8)
            assert_allclose(scaled, p(expected), type_test="tensor")

    def test_image_scale_intensity_range_none_clip(self):
        scaler = ScaleIntensityRange(a_min=20, a_max=108, b_min=None, b_max=80, clip=True, dtype=np.uint8)
        for p in TEST_NDARRAYS:
            scaled = scaler(p(self.imt))
            self.assertTrue(scaled.dtype, np.uint8)
            expected = (np.clip((self.imt - 20) / 88, None, 80)).astype(np.uint8)
            assert_allclose(scaled, p(expected), type_test="tensor")


if __name__ == "__main__":
    unittest.main()
