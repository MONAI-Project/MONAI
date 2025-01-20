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
from copy import deepcopy

from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.transforms import KSpaceSpikeNoise, RandKSpaceSpikeNoise
from monai.utils.misc import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for shape in ((128, 64), (64, 48, 80)):
    for p in TEST_NDARRAYS:
        for channel_wise in (True, False):
            TESTS.append((shape, p, channel_wise))


class TestRandKSpaceSpikeNoise(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, im_type):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        im = create_test_image(*im_shape, rad_max=20, noise_max=0.0, num_seg_classes=5)[0][None]
        return im_type(im)

    @parameterized.expand(TESTS)
    def test_0_prob(self, im_shape, im_type, channel_wise):
        im = self.get_data(im_shape, im_type)
        intensity_range = [14, 15]
        t = RandKSpaceSpikeNoise(0.0, intensity_range, channel_wise)
        out = t(im)
        assert_allclose(out, im, type_test="tensor")

    @parameterized.expand(TESTS)
    def test_1_prob(self, im_shape, im_type, channel_wise):
        im = self.get_data(im_shape, im_type)
        intensity_range = [14, 14]
        t = RandKSpaceSpikeNoise(1.0, intensity_range, channel_wise)
        out = t(im)
        base_t = KSpaceSpikeNoise(t.sampled_locs, [14])
        out = out - base_t(im)
        assert_allclose(out, im * 0, type_test="tensor")

    @parameterized.expand(TESTS)
    def test_same_result(self, im_shape, im_type, channel_wise):
        im = self.get_data(im_shape, im_type)
        intensity_range = [14, 15]
        t = RandKSpaceSpikeNoise(0.0, intensity_range, channel_wise)
        t.set_random_state(42)
        out1 = t(deepcopy(im))
        t.set_random_state(42)
        out2 = t(deepcopy(im))
        assert_allclose(out1, out2, type_test="tensor")

    @parameterized.expand(TESTS)
    def test_intensity(self, im_shape, im_type, channel_wise):
        im = self.get_data(im_shape, im_type)
        intensity_range = [14, 14.1]
        t = RandKSpaceSpikeNoise(1.0, intensity_range, channel_wise)
        _ = t(deepcopy(im))
        self.assertGreaterEqual(t.sampled_k_intensity[0], 14)
        self.assertLessEqual(t.sampled_k_intensity[0], 14.1)

    @parameterized.expand(TESTS)
    def test_default_intensity(self, im_shape, im_type, channel_wise):
        im = self.get_data(im_shape, im_type)
        t = RandKSpaceSpikeNoise(1.0, intensity_range=None, channel_wise=channel_wise)
        out = t(deepcopy(im))
        self.assertEqual(out.shape, im.shape)


if __name__ == "__main__":
    unittest.main()
