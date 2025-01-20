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

from parameterized import parameterized

from monai.transforms import ClipIntensityPercentilesd
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile
from monai.utils.type_conversion import convert_to_tensor
from tests.transforms.test_clip_intensity_percentiles import test_hard_clip_func, test_soft_clip_func
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, NumpyImageTestCase3D, assert_allclose


class TestClipIntensityPercentilesd2D(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_hard_clipping_two_sided(self, p):
        key = "img"
        hard_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=5)
        im = p(self.imt)
        result = hard_clipper({key: im})
        expected = test_hard_clip_func(im, 5, 95)
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_hard_clipping_one_sided_high(self, p):
        key = "img"
        hard_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=None)
        im = p(self.imt)
        result = hard_clipper({key: im})
        expected = test_hard_clip_func(im, 0, 95)
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_hard_clipping_one_sided_low(self, p):
        key = "img"
        hard_clipper = ClipIntensityPercentilesd(keys=[key], upper=None, lower=5)
        im = p(self.imt)
        result = hard_clipper({key: im})
        expected = test_hard_clip_func(im, 5, 100)
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_soft_clipping_two_sided(self, p):
        key = "img"
        soft_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=5, sharpness_factor=1.0)
        im = p(self.imt)
        result = soft_clipper({key: im})
        expected = test_soft_clip_func(im, 5, 95)
        # the rtol is set to 1e-4 because the logaddexp function used in softplus is not stable accross torch and numpy
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_soft_clipping_one_sided_high(self, p):
        key = "img"
        soft_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=None, sharpness_factor=1.0)
        im = p(self.imt)
        result = soft_clipper({key: im})
        expected = test_soft_clip_func(im, None, 95)
        # the rtol is set to 1e-4 because the logaddexp function used in softplus is not stable accross torch and numpy
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_soft_clipping_one_sided_low(self, p):
        key = "img"
        soft_clipper = ClipIntensityPercentilesd(keys=[key], upper=None, lower=5, sharpness_factor=1.0)
        im = p(self.imt)
        result = soft_clipper({key: im})
        expected = test_soft_clip_func(im, 5, None)
        # the rtol is set to 1e-4 because the logaddexp function used in softplus is not stable accross torch and numpy
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, p):
        key = "img"
        clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=5, channel_wise=True)
        im = p(self.imt)
        result = clipper({key: im})
        im_t = convert_to_tensor(self.imt)
        for i, c in enumerate(im_t):
            lower, upper = percentile(c, (5, 95))
            expected = clip(c, lower, upper)
            assert_allclose(result[key][i], p(expected), type_test="tensor", rtol=1e-3, atol=0)

    def test_ill_sharpness_factor(self):
        key = "img"
        with self.assertRaises(ValueError):
            ClipIntensityPercentilesd(keys=[key], upper=95, lower=5, sharpness_factor=0.0)

    def test_ill_lower_percentile(self):
        key = "img"
        with self.assertRaises(ValueError):
            ClipIntensityPercentilesd(keys=[key], upper=None, lower=-1)

    def test_ill_upper_percentile(self):
        key = "img"
        with self.assertRaises(ValueError):
            ClipIntensityPercentilesd(keys=[key], upper=101, lower=None)

    def test_ill_percentiles(self):
        key = "img"
        with self.assertRaises(ValueError):
            ClipIntensityPercentilesd(keys=[key], upper=95, lower=96)

    def test_ill_both_none(self):
        key = "img"
        with self.assertRaises(ValueError):
            ClipIntensityPercentilesd(keys=[key], upper=None, lower=None)


class TestClipIntensityPercentilesd3D(NumpyImageTestCase3D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_hard_clipping_two_sided(self, p):
        key = "img"
        hard_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=5)
        im = p(self.imt)
        result = hard_clipper({key: im})
        expected = test_hard_clip_func(im, 5, 95)
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_hard_clipping_one_sided_high(self, p):
        key = "img"
        hard_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=None)
        im = p(self.imt)
        result = hard_clipper({key: im})
        expected = test_hard_clip_func(im, 0, 95)
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_hard_clipping_one_sided_low(self, p):
        key = "img"
        hard_clipper = ClipIntensityPercentilesd(keys=[key], upper=None, lower=5)
        im = p(self.imt)
        result = hard_clipper({key: im})
        expected = test_hard_clip_func(im, 5, 100)
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_soft_clipping_two_sided(self, p):
        key = "img"
        soft_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=5, sharpness_factor=1.0)
        im = p(self.imt)
        result = soft_clipper({key: im})
        expected = test_soft_clip_func(im, 5, 95)
        # the rtol is set to 1e-4 because the logaddexp function used in softplus is not stable accross torch and numpy
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_soft_clipping_one_sided_high(self, p):
        key = "img"
        soft_clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=None, sharpness_factor=1.0)
        im = p(self.imt)
        result = soft_clipper({key: im})
        expected = test_soft_clip_func(im, None, 95)
        # the rtol is set to 1e-4 because the logaddexp function used in softplus is not stable accross torch and numpy
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_soft_clipping_one_sided_low(self, p):
        key = "img"
        soft_clipper = ClipIntensityPercentilesd(keys=[key], upper=None, lower=5, sharpness_factor=1.0)
        im = p(self.imt)
        result = soft_clipper({key: im})
        expected = test_soft_clip_func(im, 5, None)
        # the rtol is set to 1e-6 because the logaddexp function used in softplus is not stable accross torch and numpy
        assert_allclose(result[key], p(expected), type_test="tensor", rtol=1e-4, atol=0)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, p):
        key = "img"
        clipper = ClipIntensityPercentilesd(keys=[key], upper=95, lower=5, channel_wise=True)
        im = p(self.imt)
        result = clipper({key: im})
        im_t = convert_to_tensor(im)
        for i, c in enumerate(im_t):
            lower, upper = percentile(c, (5, 95))
            expected = clip(c, lower, upper)
            assert_allclose(result[key][i], p(expected), type_test="tensor", rtol=1e-4, atol=0)


if __name__ == "__main__":
    unittest.main()
