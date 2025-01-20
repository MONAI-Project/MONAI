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
from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.transforms import GibbsNoised
from monai.utils.misc import set_determinism
from monai.utils.module import optional_import
from tests.util import TEST_NDARRAYS, assert_allclose

_, has_torch_fft = optional_import("torch.fft", name="fftshift")

TEST_CASES = []
for shape in ((128, 64), (64, 48, 80)):
    for input_type in TEST_NDARRAYS if has_torch_fft else [np.array]:
        TEST_CASES.append((shape, input_type))
KEYS = ["im", "label"]


class TestGibbsNoised(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, input_type):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        ims = create_test_image(*im_shape, rad_max=20, noise_max=0.0, num_seg_classes=5)
        return {k: input_type(deepcopy(v)) for k, v in zip(KEYS, ims)}

    @parameterized.expand(TEST_CASES)
    def test_same_result(self, im_shape, input_type):
        data = self.get_data(im_shape, input_type)
        alpha = 0.8
        t = GibbsNoised(KEYS, alpha)
        out1 = t(deepcopy(data))
        out2 = t(deepcopy(data))
        for k in KEYS:
            assert_allclose(out1[k], out2[k], rtol=1e-7, atol=0, type_test="tensor")

    @parameterized.expand(TEST_CASES)
    def test_identity(self, im_shape, input_type):
        data = self.get_data(im_shape, input_type)
        alpha = 0.0
        t = GibbsNoised(KEYS, alpha)
        out = t(deepcopy(data))
        for k in KEYS:
            assert_allclose(out[k], data[k], atol=1e-2, type_test="tensor")

    @parameterized.expand(TEST_CASES)
    def test_alpha_1(self, im_shape, input_type):
        data = self.get_data(im_shape, input_type)
        alpha = 1.0
        t = GibbsNoised(KEYS, alpha)
        out = t(deepcopy(data))
        for k in KEYS:
            assert_allclose(out[k], 0.0 * data[k], atol=1e-2, type_test="tensor")

    @parameterized.expand(TEST_CASES)
    def test_dict_matches(self, im_shape, input_type):
        data = self.get_data(im_shape, input_type)
        data = {KEYS[0]: deepcopy(data[KEYS[0]]), KEYS[1]: deepcopy(data[KEYS[0]])}
        alpha = 1.0
        t = GibbsNoised(KEYS, alpha)
        out = t(deepcopy(data))
        assert_allclose(out[KEYS[0]], out[KEYS[1]], rtol=1e-7, atol=0, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
