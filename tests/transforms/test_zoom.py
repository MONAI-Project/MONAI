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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Zoom
from monai.transforms.lazy.functional import apply_pending
from tests.util import (
    DEFAULT_TEST_AFFINE,
    TEST_NDARRAYS_ALL,
    NumpyImageTestCase2D,
    assert_allclose,
    test_local_inversion,
)

VALID_CASES = [
    (1.5, "nearest", True),
    (1.5, "nearest", False),
    (0.8, "bilinear"),
    (0.8, 1),
    (0.8, "area"),
    (1.5, "nearest", False, True),
    (0.8, "area", False, True),
]

INVALID_CASES = [((None, None), "bilinear", TypeError), ((0.9, 0.9), "s", ValueError)]


class TestZoom(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_pending_ops(self, zoom, mode, align_corners=False, keep_size=False):
        im = MetaTensor(self.imt[0], meta={"a": "b", "affine": DEFAULT_TEST_AFFINE})
        zoom_fn = Zoom(
            zoom=zoom, mode="bilinear", keep_size=keep_size, dtype=torch.float64, align_corners=align_corners
        )
        # non-lazy
        expected = zoom_fn(im)
        self.assertIsInstance(expected, MetaTensor)
        # lazy
        zoom_fn.lazy = True
        pending_result = zoom_fn(im)
        self.assertIsInstance(pending_result, MetaTensor)
        assert_allclose(pending_result.peek_pending_affine(), expected.affine)
        assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
        overrides = {"mode": "bilinear", "dtype": np.float64, "align_corners": align_corners}
        result = apply_pending(pending_result, overrides=overrides)[0]
        # compare
        match_ratio = np.sum(np.isclose(result, expected)) / np.prod(result.shape)
        self.assertGreater(match_ratio, 0.95)

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, zoom, mode, *_):
        for p in TEST_NDARRAYS_ALL:
            zoom_fn = Zoom(zoom=zoom, mode=mode, keep_size=False)
            im = p(self.imt[0])
            zoomed = zoom_fn(im)
            test_local_inversion(zoom_fn, zoomed, im)
            _order = 0
            if mode == 1 or mode.endswith("linear"):
                _order = 1
            expected = []
            for channel in self.imt[0]:
                expected.append(zoom_scipy(channel, zoom=zoom, mode="nearest", order=_order, prefilter=False))
            expected = np.stack(expected).astype(np.float32)
            assert_allclose(zoomed, p(expected), atol=1.0, type_test=False)

    def test_keep_size(self):
        for p in TEST_NDARRAYS_ALL:
            zoom_fn = Zoom(zoom=[0.6, 0.6], keep_size=True, align_corners=True)
            im = p(self.imt[0])
            zoomed = zoom_fn(im, mode="bilinear")
            assert_allclose(zoomed.shape, self.imt.shape[1:], type_test=False)
            test_local_inversion(zoom_fn, zoomed, im)

            zoom_fn = Zoom(zoom=[1.3, 1.3], keep_size=True)
            im = p(self.imt[0])
            zoomed = zoom_fn(im)
            assert_allclose(zoomed.shape, self.imt.shape[1:], type_test=False)
            test_local_inversion(zoom_fn, zoomed, p(self.imt[0]))

            set_track_meta(False)
            rotated = zoom_fn(im)
            self.assertNotIsInstance(rotated, MetaTensor)
            np.testing.assert_allclose(zoomed.shape, self.imt.shape[1:])
            set_track_meta(True)

    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, zoom, mode, raises):
        for p in TEST_NDARRAYS_ALL:
            with self.assertRaises(raises):
                zoom_fn = Zoom(zoom=zoom, mode=mode)
                zoom_fn(p(self.imt[0]))

    def test_padding_mode(self):
        for p in TEST_NDARRAYS_ALL:
            zoom_fn = Zoom(zoom=0.5, mode="nearest", padding_mode="constant", keep_size=True)
            test_data = p([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
            zoomed = zoom_fn(test_data)
            expected = p([[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
            assert_allclose(zoomed, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
