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

from monai.data.meta_tensor import MetaTensor
from monai.transforms import ResizeWithPadOrCrop
from monai.transforms.lazy.functional import apply_pending
from tests.util import TEST_NDARRAYS_ALL, assert_allclose, pytorch_after

TEST_CASES = [
    [{"spatial_size": [15, 8, 8], "mode": "constant"}, (3, 8, 8, 4), (3, 15, 8, 8), True],
    [{"spatial_size": [15, 4, -1], "mode": "constant"}, (3, 8, 8, 4), (3, 15, 4, 4), True],
    [
        {"spatial_size": [15, 4, -1], "mode": "reflect" if pytorch_after(1, 11) else "constant"},
        (3, 8, 8, 4),
        (3, 15, 4, 4),
        True,
    ],
    [
        {"spatial_size": [-1, -1, -1], "mode": "reflect" if pytorch_after(1, 11) else "constant"},
        (3, 8, 8, 4),
        (3, 8, 8, 4),
        True,
    ],
    [
        {"spatial_size": [15, 4, 8], "mode": "constant", "method": "end", "constant_values": 1},
        (3, 8, 8, 4),
        (3, 15, 4, 8),
        True,
    ],
]
TESTS_PENDING_MODE = {"constant": "zeros", "edge": "border", "reflect": "reflection"}


class TestResizeWithPadOrCrop(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_pad_shape(self, input_param, input_shape, expected_shape, _):
        for p in TEST_NDARRAYS_ALL:
            if isinstance(p(0), torch.Tensor) and (
                "constant_values" in input_param or input_param["mode"] == "reflect"
            ):
                continue
            padcropper = ResizeWithPadOrCrop(**input_param)
            result = padcropper(p(np.zeros(input_shape)))
            np.testing.assert_allclose(result.shape, expected_shape)
            result = padcropper(p(np.zeros(input_shape)), mode="constant")
            np.testing.assert_allclose(result.shape, expected_shape)
            self.assertIsInstance(result, MetaTensor)
            self.assertEqual(len(result.applied_operations), 1)
            inv = padcropper.inverse(result)
            self.assertTupleEqual(inv.shape, input_shape)
            self.assertIsInstance(inv, MetaTensor)
            self.assertEqual(inv.applied_operations, [])

    @parameterized.expand(TEST_CASES)
    def test_pending_ops(self, input_param, input_shape, _expected_data, align_corners):
        for p in TEST_NDARRAYS_ALL:
            # grid sample only support constant value to be zero
            if "constant_values" in input_param and input_param["constant_values"] != 0:
                continue
            padcropper = ResizeWithPadOrCrop(**input_param)
            image = p(np.zeros(input_shape))
            # non-lazy
            expected = padcropper(image)
            self.assertIsInstance(expected, MetaTensor)
            # lazy
            padcropper.lazy = True
            pending_result = padcropper(image)
            self.assertIsInstance(pending_result, MetaTensor)
            assert_allclose(pending_result.peek_pending_affine(), expected.affine)
            assert_allclose(pending_result.peek_pending_shape(), expected.shape[1:])
            # only support nearest
            overrides = {
                "mode": "nearest",
                "padding_mode": TESTS_PENDING_MODE[input_param["mode"]],
                "align_corners": align_corners,
            }
            result = apply_pending(pending_result, overrides=overrides)[0]
            # compare
            assert_allclose(result, expected, rtol=1e-5)
            inverted = padcropper.inverse(result)
            self.assertEqual(inverted.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
