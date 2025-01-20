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
from monai.transforms import KSpaceSpikeNoise
from monai.utils.misc import set_determinism
from tests.util import TEST_NDARRAYS

TESTS = []
for shape in ((128, 64), (64, 48, 80)):
    for p in TEST_NDARRAYS:
        for intensity in [10, None]:
            TESTS.append((shape, p, intensity))


class TestKSpaceSpikeNoise(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, im_type):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        im, _ = create_test_image(*im_shape, rad_max=20, noise_max=0.0, num_seg_classes=5)
        return im_type(im[None])

    @parameterized.expand(TESTS)
    def test_same_result(self, im_shape, im_type, k_intensity):
        im = self.get_data(im_shape, im_type)
        loc = [0, int(im.shape[1] / 2), 0] if len(im_shape) == 2 else [0, int(im.shape[1] / 2), 0, 0]
        t = KSpaceSpikeNoise(loc, k_intensity)

        out1 = t(deepcopy(im))
        out2 = t(deepcopy(im))

        if isinstance(out1, torch.Tensor):
            out1 = out1.cpu()
            out2 = out2.cpu()

        np.testing.assert_allclose(out1, out2)

    @parameterized.expand(TESTS)
    def test_highlighted_kspace_pixel(self, im_shape, as_tensor_input, k_intensity):
        im = self.get_data(im_shape, as_tensor_input)
        loc = [0, int(im.shape[1] / 2), 0] if len(im_shape) == 2 else [0, int(im.shape[1] / 2), 0, 0]
        t = KSpaceSpikeNoise(loc, k_intensity)
        out = t(im)

        if isinstance(out, torch.Tensor):
            out = out.cpu()

        if k_intensity is not None:
            n_dims = len(im_shape)
            out_k = fftshift(fftn(out, axes=tuple(range(-n_dims, 0))), axes=tuple(range(-n_dims, 0)))
            log_mag = np.log(np.absolute(out_k))
            np.testing.assert_allclose(k_intensity, log_mag[tuple(loc)], 1e-4)


if __name__ == "__main__":
    unittest.main()
