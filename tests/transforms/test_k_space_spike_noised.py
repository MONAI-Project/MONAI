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

import numpy as np
import torch
from numpy.fft import fftn, fftshift
from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.transforms import KSpaceSpikeNoised
from monai.utils.misc import set_determinism
from tests.util import TEST_NDARRAYS, assert_allclose

TESTS = []
for shape in ((128, 64), (64, 48, 80)):
    for p in TEST_NDARRAYS:
        TESTS.append((shape, p))

KEYS = ["image", "label"]


class TestKSpaceSpikeNoised(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, im_type):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        ims = create_test_image(*im_shape, rad_max=20, noise_max=0.0, num_seg_classes=5)
        ims = [im_type(im[None]) for im in ims]
        return dict(zip(KEYS, ims))

    @parameterized.expand(TESTS)
    def test_same_result(self, im_shape, im_type):
        data = self.get_data(im_shape, im_type)
        loc = [0] + [int(im_shape[i] / 2) for i in range(len(im_shape))]
        k_intensity = 10

        t = KSpaceSpikeNoised(KEYS, loc, k_intensity)
        out1 = t(deepcopy(data))
        out2 = t(deepcopy(data))

        for k in KEYS:
            if isinstance(out1[k], torch.Tensor):
                out1[k] = out1[k].cpu()
                out2[k] = out2[k].cpu()
            np.testing.assert_allclose(out1[k], out2[k])

    @parameterized.expand(TESTS)
    def test_highlighted_kspace_pixel(self, im_shape, im_type):
        data = self.get_data(im_shape, im_type)
        loc = [0] + [int(im_shape[i] / 2) for i in range(len(im_shape))]
        k_intensity = 10

        t = KSpaceSpikeNoised(KEYS, loc, k_intensity)
        out = t(data)

        for k in KEYS:
            if isinstance(out[k], torch.Tensor):
                out[k] = out[k].cpu()

            n_dims = len(im_shape)
            out_k = fftshift(fftn(out[k], axes=tuple(range(-n_dims, 0))), axes=tuple(range(-n_dims, 0)))
            log_mag = np.log(np.absolute(out_k))
            np.testing.assert_allclose(k_intensity, log_mag[tuple(loc)], 1e-1)

    @parameterized.expand(TESTS)
    def test_dict_matches(self, im_shape, im_type):
        data = self.get_data(im_shape, im_type)
        # use same image for both dictionary entries to check same trans is applied to them
        data = {KEYS[0]: deepcopy(data[KEYS[0]]), KEYS[1]: deepcopy(data[KEYS[0]])}
        loc = [0] + [int(im_shape[i] / 2) for i in range(len(im_shape))]
        k_intensity = 10

        t = KSpaceSpikeNoised(KEYS, loc, k_intensity)
        out = t(deepcopy(data))
        assert_allclose(out[KEYS[0]], out[KEYS[1]], type_test=False)


if __name__ == "__main__":
    unittest.main()
