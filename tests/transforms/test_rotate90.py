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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Affine, Rotate90
from monai.transforms.lazy.functional import apply_pending
from monai.utils import optional_import
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import (
    TEST_NDARRAYS_ALL,
    NumpyImageTestCase2D,
    NumpyImageTestCase3D,
    assert_allclose,
    test_local_inversion,
)


class TestRotate90(NumpyImageTestCase2D):
    def test_rotate90_default(self):
        rotate = Rotate90()
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            set_track_meta(True)
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")
            set_track_meta(False)
            rotated = rotate(im)
            self.assertNotIsInstance(rotated, MetaTensor)
            set_track_meta(True)

    def test_k(self):
        rotate = Rotate90(k=2)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_spatial_axes(self):
        rotate = Rotate90(spatial_axes=(0, -1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, -1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_prob_k_spatial_axes(self):
        rotate = Rotate90(k=2, spatial_axes=(0, 1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")


class TestRotate903d(NumpyImageTestCase3D):
    def test_rotate90_default(self):
        rotate = Rotate90()
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_k(self):
        rotate = Rotate90(k=2)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_spatial_axes(self):
        rotate = Rotate90(spatial_axes=(0, -1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, -1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_prob_k_spatial_axes(self):
        rotate = Rotate90(k=2, spatial_axes=(0, 1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            call_param = {"img": im}
            rotated = rotate(**call_param)

            # test lazy
            test_resampler_lazy(rotate, rotated, call_param=call_param)
            rotate.lazy = False

            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")


@unittest.skipUnless(optional_import("scipy")[1], "Requires scipy library.")
class TestRot90Consistency(unittest.TestCase):
    @parameterized.expand([[2], [3], [4]])
    def test_affine_rot90(self, s):
        """s"""
        im = np.arange(int(s * s)).reshape(1, s, s).astype(float)
        mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        def method_0(im, ac):
            xform = Affine(align_corners=ac, affine=mat, image_only=True, spatial_size=s)
            xform.lazy = True
            out = xform(im)
            out = apply_pending(out, overrides={"padding_mode": "border", "align_corners": ac})[0]
            return out

        def method_1(im, ac):
            xform = Affine(align_corners=ac, affine=mat, image_only=True, spatial_size=s)
            xform.lazy = True
            out = xform(im)
            out = apply_pending(out, overrides={"mode": 1, "padding_mode": "nearest", "align_corners": ac})[0]
            return out

        def method_2(im, ac):
            xform = Affine(align_corners=ac, affine=mat, padding_mode="border", image_only=True, spatial_size=s)
            out = xform(im)
            return out

        def method_3(im, ac):
            xform = Affine(
                align_corners=ac, affine=mat, mode=1, padding_mode="nearest", image_only=True, spatial_size=s
            )
            out = xform(im)
            return out

        for call in (method_0, method_1, method_2, method_3):
            for ac in (False, True):
                out = call(im, ac)
                ref = Rotate90()(im)
                assert_allclose(out, ref, rtol=1e-4, atol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
