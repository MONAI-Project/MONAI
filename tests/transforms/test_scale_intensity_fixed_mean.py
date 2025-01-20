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

from monai.transforms import ScaleIntensityFixedMean
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestScaleIntensityFixedMean(NumpyImageTestCase2D):
    def test_factor_scale(self):
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensityFixedMean(factor=0.1, fixed_mean=False)
            result = scaler(p(self.imt))
            expected = p((self.imt * (1 + 0.1)).astype(np.float32))
            assert_allclose(result, p(expected), type_test="tensor", rtol=1e-7, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_preserve_range(self, p):
        for channel_wise in [False, True]:
            factor = 0.9
            scaler = ScaleIntensityFixedMean(
                factor=factor, preserve_range=True, channel_wise=channel_wise, fixed_mean=False
            )
            im = p(self.imt)
            result = scaler(im)

            if False:  # channel_wise:
                out = []
                for d in im:
                    clip_min = d.min()
                    clip_max = d.max()
                    d = (1 + factor) * d
                    d[d < clip_min] = clip_min
                    d[d > clip_max] = clip_max
                    out.append(d)
                expected = p(out)
            else:
                clip_min = im.min()
                clip_max = im.max()
                im = (1 + factor) * im
                im[im < clip_min] = clip_min
                im[im > clip_max] = clip_max
                expected = im
            assert_allclose(result, expected, type_test="tensor", atol=1e-7)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_fixed_mean(self, p):
        for channel_wise in [False, True]:
            factor = 0.9
            scaler = ScaleIntensityFixedMean(factor=factor, fixed_mean=True, channel_wise=channel_wise)
            im = p(self.imt)
            result = scaler(im)
            mn = im.mean()
            im = im - mn
            expected = (1 + factor) * im
            expected = expected + mn
            assert_allclose(result, expected, type_test="tensor", atol=1e-7)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_fixed_mean_preserve_range(self, p):
        for channel_wise in [False, True]:
            factor = 0.9
            scaler = ScaleIntensityFixedMean(
                factor=factor, preserve_range=True, fixed_mean=True, channel_wise=channel_wise
            )
            im = p(self.imt)
            clip_min = im.min()
            clip_max = im.max()
            result = scaler(im)
            mn = im.mean()
            im = im - mn
            expected = (1 + factor) * im
            expected = expected + mn
            expected[expected < clip_min] = clip_min
            expected[expected > clip_max] = clip_max
            assert_allclose(result, expected, type_test="tensor", atol=1e-7)


if __name__ == "__main__":
    unittest.main()
