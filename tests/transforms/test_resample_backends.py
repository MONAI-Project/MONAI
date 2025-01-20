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

from monai.config import USE_COMPILED
from monai.data import MetaTensor
from monai.transforms import Resample
from monai.transforms.utils import create_grid
from monai.utils import GridSampleMode, GridSamplePadMode, NdimageMode, SplineMode, convert_to_numpy
from tests.util import SkipIfBeforePyTorchVersion, assert_allclose, is_tf32_env

_rtol = 1e-3 if is_tf32_env() else 1e-4

TEST_IDENTITY = []
for interp in GridSampleMode if not USE_COMPILED else ("nearest", "bilinear"):  # type: ignore
    for pad in GridSamplePadMode:
        for p in (np.float32, np.float64):
            for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
                TEST_IDENTITY.append([dict(device=device), p, interp, pad, (1, 3, 4)])
                if interp != "bicubic":
                    TEST_IDENTITY.append([dict(device=device), p, interp, pad, (1, 3, 5, 8)])
for interp_s in SplineMode if not USE_COMPILED else []:  # type: ignore
    for pad_s in NdimageMode:
        for p_s in (int, float, np.float32, np.float64):
            for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
                TEST_IDENTITY.append([dict(device=device), p_s, interp_s, pad_s, (1, 20, 21)])
                TEST_IDENTITY.append([dict(device=device), p_s, interp_s, pad_s, (1, 21, 23, 24)])


@SkipIfBeforePyTorchVersion((1, 9, 1))
class TestResampleBackends(unittest.TestCase):
    @parameterized.expand(TEST_IDENTITY)
    def test_resample_identity(self, input_param, im_type, interp, padding, input_shape):
        """test resampling of an identity grid with padding 2, im_type, interp, padding, input_shape"""
        xform = Resample(dtype=im_type, **input_param)
        n_elem = np.prod(input_shape)
        img = convert_to_numpy(np.arange(n_elem).reshape(input_shape), dtype=im_type)
        grid = create_grid(input_shape[1:], homogeneous=True, backend="numpy")
        grid_p = np.stack([np.pad(g, 2, "constant") for g in grid])  # testing pad
        output = xform(img=img, grid=grid_p, mode=interp, padding_mode=padding)
        self.assertTrue(not torch.any(torch.isinf(output) | torch.isnan(output)))
        self.assertIsInstance(output, MetaTensor)
        slices = [slice(None)]
        slices.extend([slice(2, -2) for _ in img.shape[1:]])
        output_c = output[slices]
        assert_allclose(output_c, img, rtol=_rtol, atol=1e-3, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
