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
import torch
from parameterized import parameterized
from scipy.ndimage import zoom as zoom_scipy

from monai.config import USE_COMPILED
from monai.transforms import RandZoom
from monai.utils import InterpolateMode
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

VALID_CASES = [
    (0.8, 1.2, "nearest", False),
    (0.8, 1.2, InterpolateMode.NEAREST, False),
    (0.8, 1.2, InterpolateMode.BILINEAR, False, True),
    (0.8, 1.2, InterpolateMode.BILINEAR, False, False),
]


class TestRandZoom(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, min_zoom, max_zoom, mode, keep_size, align_corners=None):
        for p in TEST_NDARRAYS_ALL:
            init_param = {
                "prob": 1.0,
                "min_zoom": min_zoom,
                "max_zoom": max_zoom,
                "mode": mode,
                "keep_size": keep_size,
                "dtype": torch.float64,
                "align_corners": align_corners,
            }
            random_zoom = RandZoom(**init_param)
            random_zoom.set_random_state(1234)
            im = p(self.imt[0])
            call_param = {"img": im}
            zoomed = random_zoom(**call_param)

            # test lazy
            # TODO: temporarily skip "nearest" test
            if mode == InterpolateMode.BILINEAR:
                test_resampler_lazy(
                    random_zoom, zoomed, init_param, call_param, seed=1234, atol=1e-4 if USE_COMPILED else 1e-6
                )

            test_local_inversion(random_zoom, zoomed, im)
            expected = [
                zoom_scipy(channel, zoom=random_zoom._zoom, mode="nearest", order=0, prefilter=False)
                for channel in self.imt[0]
            ]

            expected = np.stack(expected).astype(np.float32)
            assert_allclose(zoomed, p(expected), atol=1.0, type_test=False)

    def test_keep_size(self):
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            random_zoom = RandZoom(prob=1.0, min_zoom=0.6, max_zoom=0.7, keep_size=True)
            random_zoom.set_random_state(12)
            zoomed = random_zoom(im)
            test_local_inversion(random_zoom, zoomed, im)
            self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape[1:]))
            zoomed = random_zoom(im)
            self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape[1:]))
            zoomed = random_zoom(im)
            self.assertTrue(np.array_equal(zoomed.shape, self.imt.shape[1:]))
            random_zoom.prob = 0.0
            self.assertEqual(random_zoom(im).dtype, torch.float32)

    @parameterized.expand(
        [("no_min_zoom", None, 1.1, "bilinear", TypeError), ("invalid_mode", 0.9, 1.1, "s", ValueError)]
    )
    def test_invalid_inputs(self, _, min_zoom, max_zoom, mode, raises):
        for p in TEST_NDARRAYS_ALL:
            with self.assertRaises(raises):
                random_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, mode=mode)
                random_zoom(p(self.imt[0]))

    def test_auto_expand_3d(self):
        for p in TEST_NDARRAYS_ALL:
            random_zoom = RandZoom(prob=1.0, min_zoom=[0.8, 0.7], max_zoom=[1.2, 1.3], mode="nearest", keep_size=False)
            random_zoom.set_random_state(1234)
            test_data = p(np.random.randint(0, 2, size=[2, 2, 3, 4]))
            zoomed = random_zoom(test_data)
            assert_allclose(random_zoom._zoom, (1.048844, 1.048844, 0.962637), atol=1e-2, type_test=False)
            assert_allclose(zoomed.shape, (2, 2, 3, 3), type_test=False)


if __name__ == "__main__":
    unittest.main()
