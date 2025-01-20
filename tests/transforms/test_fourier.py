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

from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.transforms import Fourier
from monai.utils.misc import set_determinism
from tests.util import SkipIfBeforePyTorchVersion, SkipIfNoModule

TEST_CASES = [((128, 64),), ((64, 48, 80),)]


@SkipIfBeforePyTorchVersion((1, 8))
@SkipIfNoModule("torch.fft")
class TestFourier(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(img_shape):
        create_test_image = create_test_image_2d if len(img_shape) == 2 else create_test_image_3d
        im = create_test_image(*img_shape, num_objs=4, rad_max=20, noise_max=0.0, num_seg_classes=5)[0][None]
        return torch.Tensor(im)

    @parameterized.expand(TEST_CASES)
    def test_forward(self, img_shape):
        n_dims = len(img_shape[1:])
        x = self.get_data(img_shape)
        t = Fourier()
        out = t.shift_fourier(x, n_dims)

        expect = torch.fft.fftshift(torch.fft.fftn(x, dim=tuple(range(-n_dims, 0))), dim=tuple(range(-n_dims, 0)))

        np.testing.assert_allclose(out, expect)

    @parameterized.expand(TEST_CASES)
    def test_backward(self, img_shape):
        n_dims = len(img_shape[1:])
        x = self.get_data(img_shape)
        t = Fourier()
        out = t.inv_shift_fourier(x, n_dims)

        expect = torch.fft.ifftn(
            torch.fft.ifftshift(x, dim=tuple(range(-n_dims, 0))), dim=tuple(range(-n_dims, 0))
        ).real

        np.testing.assert_allclose(out, expect)


if __name__ == "__main__":
    unittest.main()
