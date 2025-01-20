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
import skimage.transform
import torch
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Resize
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import (
    TEST_NDARRAYS_ALL,
    NumpyImageTestCase2D,
    SkipIfAtLeastPyTorchVersion,
    assert_allclose,
    is_tf32_env,
    pytorch_after,
)

TEST_CASE_0 = [{"spatial_size": 15}, (6, 10, 15)]

TEST_CASE_1 = [{"spatial_size": 15, "mode": "area"}, (6, 10, 15)]

TEST_CASE_2 = [{"spatial_size": 6, "mode": "trilinear", "align_corners": True}, (2, 4, 6)]

TEST_CASE_2_1 = [{"spatial_size": 6, "mode": 1, "align_corners": True}, (2, 4, 6)]

TEST_CASE_3 = [{"spatial_size": 15, "anti_aliasing": True}, (6, 10, 15)]

TEST_CASE_4 = [{"spatial_size": 6, "anti_aliasing": True, "anti_aliasing_sigma": 2.0}, (2, 4, 6)]

diff_t = 0.3 if is_tf32_env() else 0.2


class TestResize(NumpyImageTestCase2D):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resize = Resize(spatial_size=(128, 128, 3), mode="order")
            resize(self.imt[0])

        with self.assertRaises(ValueError):
            resize = Resize(spatial_size=(128,), mode="order")
            resize(self.imt[0])

    def test_unchange(self):
        resize = Resize(spatial_size=(128, 64), mode="bilinear")
        set_track_meta(False)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            result = resize(im)
            assert_allclose(im, result, type_test=False)
        set_track_meta(True)

    @parameterized.expand(
        [
            ((32, -1), "area", True),
            ((32, 32), "area", False),
            ((32, 32, 32), "trilinear", True),
            ((256, 256), "bilinear", False),
            ((256, 256), "nearest-exact" if pytorch_after(1, 11) else "nearest", False),
            ((128, 128), "nearest", False),
            ((128, 64), "area", True),  # already in a good shape
        ]
    )
    def test_correct_results(self, spatial_size, mode, anti_aliasing):
        """resize 'spatial_size' and 'mode'"""
        init_param = {"spatial_size": spatial_size, "mode": mode, "anti_aliasing": anti_aliasing, "dtype": np.float64}
        resize = Resize(**init_param)
        _order = 0
        if mode.endswith("linear"):
            _order = 1
        if spatial_size == (32, -1):
            spatial_size = (32, 64)

        expected = [
            skimage.transform.resize(
                channel, spatial_size, order=_order, clip=False, preserve_range=False, anti_aliasing=anti_aliasing
            )
            for channel in self.imt[0]
        ]

        expected = np.stack(expected).astype(np.float32)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            out = resize(**call_param)
            if init_param["mode"] in ("bilinear", "nearest") and anti_aliasing is False:
                test_resampler_lazy(resize, out, init_param, call_param)
            if isinstance(im, MetaTensor):
                im_inv = resize.inverse(out)
                self.assertTrue(not im_inv.applied_operations)
                assert_allclose(im_inv.shape, im.shape)
                assert_allclose(im_inv.affine, im.affine, atol=1e-3, rtol=1e-3)
            if not anti_aliasing:
                assert_allclose(out, expected, type_test=False, atol=0.9)
                return
            # skimage uses reflect padding for anti-aliasing filter.
            # Our implementation reuses GaussianSmooth() as anti-aliasing filter, which uses zero padding instead.
            # Thus their results near the image boundary will be different.
            if isinstance(out, torch.Tensor):
                out = out.cpu().detach().numpy()
            good = np.sum(np.isclose(expected, out, atol=0.9))
            self.assertLessEqual(
                np.abs(good - expected.size) / float(expected.size), diff_t, f"at most {diff_t} percent mismatch "
            )

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_2_1, TEST_CASE_3, TEST_CASE_4])
    @SkipIfAtLeastPyTorchVersion((2, 2, 0))  # https://github.com/Project-MONAI/MONAI/issues/7445
    def test_longest_shape(self, input_param, expected_shape):
        input_data = np.random.randint(0, 2, size=[3, 4, 7, 10])
        input_param["size_mode"] = "longest"
        result = Resize(**input_param)(input_data)
        np.testing.assert_allclose(result.shape[1:], expected_shape)

        set_track_meta(False)
        result = Resize(**input_param)(input_data)
        self.assertNotIsInstance(result, MetaTensor)
        np.testing.assert_allclose(result.shape[1:], expected_shape)
        set_track_meta(True)

    def test_longest_infinite_decimals(self):
        resize = Resize(spatial_size=1008, size_mode="longest", mode="bilinear", align_corners=False)
        ret = resize(np.random.randint(0, 2, size=[1, 2544, 3032]))
        self.assertTupleEqual(ret.shape, (1, 846, 1008))


if __name__ == "__main__":
    unittest.main()
