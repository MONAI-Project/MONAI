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

from monai.transforms.lazy.functional import apply_pending
from monai.transforms.utils import create_rotate
from monai.utils import LazyAttr, convert_to_tensor
from tests.util import get_arange_img


def single_2d_transform_cases():
    return [
        (
            torch.as_tensor(get_arange_img((32, 32))),
            [{LazyAttr.AFFINE: create_rotate(2, np.pi / 4)}, {LazyAttr.AFFINE: create_rotate(2, -np.pi / 4)}],
            (1, 32, 32),
        ),
        (torch.as_tensor(get_arange_img((32, 32))), [create_rotate(2, np.pi / 2)], (1, 32, 32)),
        (
            torch.as_tensor(get_arange_img((16, 16))),
            [{LazyAttr.AFFINE: create_rotate(2, np.pi / 2), LazyAttr.SHAPE: (45, 45)}],
            (1, 45, 45),
        ),
    ]


class TestApply(unittest.TestCase):
    def _test_apply_impl(self, tensor, pending_transforms, expected_shape):
        result = apply_pending(tensor, pending_transforms)
        self.assertListEqual(result[1], pending_transforms)
        self.assertEqual(result[0].shape, expected_shape)

    def _test_apply_metatensor_impl(self, tensor, pending_transforms, expected_shape, pending_as_parameter):
        tensor_ = convert_to_tensor(tensor, track_meta=True)
        if pending_as_parameter:
            result, transforms = apply_pending(tensor_, pending_transforms)
        else:
            for p in pending_transforms:
                tensor_.push_pending_operation(p)
                if not isinstance(p, dict):
                    return
            result, transforms = apply_pending(tensor_)
        self.assertEqual(result.shape, expected_shape)

    SINGLE_TRANSFORM_CASES = single_2d_transform_cases()

    def test_apply_single_transform(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_impl(*case)

    def test_apply_single_transform_metatensor(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_metatensor_impl(*case, False)

    def test_apply_single_transform_metatensor_override(self):
        for case in self.SINGLE_TRANSFORM_CASES:
            self._test_apply_metatensor_impl(*case, True)


if __name__ == "__main__":
    unittest.main()
