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

import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirst, Flip, Orientation, Spacing
from monai.transforms.inverse import InvertibleTransform
from monai.utils import optional_import
from tests.util import TEST_DEVICES

_, has_nib = optional_import("nibabel")

TESTS = []
for use_compose in (False, True):
    for dtype in (torch.float32, torch.float64):
        for device in TEST_DEVICES:
            TESTS.append([use_compose, dtype, *device])


@unittest.skipUnless(has_nib, "Requires nibabel")
class TestInverseArray(unittest.TestCase):
    @staticmethod
    def get_image(dtype, device) -> MetaTensor:
        affine = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 0, 1]]).to(dtype).to(device)
        img = torch.rand((15, 16, 17)).to(dtype).to(device)
        return MetaTensor(img, affine=affine)

    @parameterized.expand(TESTS)
    def test_inverse_array(self, use_compose: bool, dtype: torch.dtype, device: torch.device):
        img: MetaTensor
        tr = Compose(
            [
                EnsureChannelFirst(channel_dim="no_channel"),
                Orientation("RAS"),
                Flip(1),
                Spacing([1.0, 1.2, 0.9], align_corners=False),
            ]
        )
        num_invertible = len([i for i in tr.transforms if isinstance(i, InvertibleTransform)])

        # forward
        img = tr(self.get_image(dtype, device))
        self.assertEqual(len(img.applied_operations), num_invertible)

        # inverse with Compose
        if use_compose:
            img = tr.inverse(img)
            self.assertEqual(len(img.applied_operations), 0)

        # inverse individually
        else:
            _tr: InvertibleTransform
            num_to_inverse = num_invertible
            for _tr in tr.transforms[::-1]:
                if isinstance(_tr, InvertibleTransform):
                    img = _tr.inverse(img)
                    num_to_inverse -= 1
                    self.assertEqual(len(img.applied_operations), num_to_inverse)


if __name__ == "__main__":
    unittest.main()
