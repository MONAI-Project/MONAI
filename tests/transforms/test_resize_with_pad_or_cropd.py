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
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import ResizeWithPadOrCropd
from monai.transforms.lazy.functional import apply_pending
from tests.transforms.test_resize_with_pad_or_crop import TESTS_PENDING_MODE
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, pytorch_after

TEST_CASES = [
    [{"keys": "img", "spatial_size": [15, 8, 8], "mode": "constant"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 8, 8)],
    [{"keys": "img", "spatial_size": [15, 4, -1], "mode": "constant"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 4, 4)],
    [
        {"keys": "img", "spatial_size": [15, 4, -1], "mode": "reflect" if pytorch_after(1, 11) else "constant"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 4),
    ],
    [
        {"keys": "img", "spatial_size": [-1, -1, -1], "mode": "reflect" if pytorch_after(1, 11) else "constant"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 8, 8, 4),
    ],
    [
        {"keys": "img", "spatial_size": [15, 4, 8], "mode": "constant", "method": "end", "constant_values": 1},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 8),
    ],
]


class TestResizeWithPadOrCropd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_pad_shape(self, input_param, input_data, expected_val):
        for p in TEST_NDARRAYS_ALL:
            if isinstance(p(0), torch.Tensor) and (
                "constant_values" in input_param or input_param["mode"] == "reflect"
            ):
                continue
            padcropper = ResizeWithPadOrCropd(**input_param)
            input_data_ = deepcopy(input_data)
            input_data_["img"] = p(input_data["img"])
            result = padcropper(input_data_)
            np.testing.assert_allclose(result["img"].shape, expected_val)
            inv = padcropper.inverse(result)
            for k in input_data_:
                self.assertTupleEqual(inv[k].shape, input_data_[k].shape)

    @parameterized.expand(TEST_CASES)
    def test_pending_ops(self, input_param, input_data, _expected_data):
        for p in TEST_NDARRAYS_ALL:
            # grid sample only support constant value to be zero
            if "constant_values" in input_param and input_param["constant_values"] != 0:
                continue
            padcropper = ResizeWithPadOrCropd(**input_param)
            input_data["img"] = p(input_data["img"])
            # non-lazy
            expected = padcropper(input_data)["img"]
            self.assertIsInstance(expected, MetaTensor)
            # lazy
            padcropper.lazy = True
            pending_result = padcropper(input_data)["img"]
            self.assertIsInstance(pending_result, MetaTensor)
            assert_allclose(pending_result.peek_pending_affine(), expected.affine)
            assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
            # only support nearest
            overrides = {
                "mode": "nearest",
                "padding_mode": TESTS_PENDING_MODE[input_param["mode"]],
                "align_corners": True,
            }
            result = apply_pending(pending_result, overrides=overrides)[0]
            # compare
            assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
