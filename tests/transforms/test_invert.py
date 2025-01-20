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

import sys
import unittest
from copy import deepcopy

import numpy as np
import torch

from monai.data import DataLoader, Dataset, MetaTensor, create_test_image_3d, decollate_batch
from monai.transforms import (
    CastToType,
    Compose,
    EnsureChannelFirst,
    Invert,
    Lambda,
    LoadImage,
    Orientation,
    RandAffine,
    RandAxisFlip,
    RandFlip,
    RandRotate,
    RandRotate90,
    RandZoom,
    ResizeWithPadOrCrop,
    Spacing,
)
from monai.utils import set_determinism
from tests.util import assert_allclose, make_nifti_image


class TestInvert(unittest.TestCase):
    def test_invert(self):
        set_determinism(seed=0)
        im_fname = make_nifti_image(create_test_image_3d(101, 100, 107, noise_max=100)[1])  # label image, discrete
        data = [im_fname for _ in range(12)]
        transform = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation("RPS"),
                Spacing(pixdim=(1.2, 1.01, 0.9), mode=1, dtype=np.float32),
                RandFlip(prob=0.5, spatial_axis=[1, 2]),
                RandAxisFlip(prob=0.5),
                RandRotate90(prob=0, spatial_axes=(1, 2)),
                RandZoom(prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
                RandRotate(prob=0.5, range_x=np.pi, mode="bilinear", align_corners=True, dtype=np.float64),
                RandAffine(prob=0.5, rotate_range=np.pi, mode="nearest"),
                ResizeWithPadOrCrop(100),
                CastToType(dtype=torch.uint8),
            ]
        )

        # num workers = 0 for mac or gpu transforms
        num_workers = 0 if sys.platform != "linux" or torch.cuda.is_available() else 2
        dataset = Dataset(data, transform=transform)
        self.assertIsInstance(transform.inverse(dataset[0]), MetaTensor)
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=1)
        inverter = Invert(transform=transform, nearest_interp=True, device="cpu", post_func=torch.as_tensor)

        for d in loader:
            d = decollate_batch(d)
            for item in d:
                orig = deepcopy(item)
                i = inverter(item)
                self.assertTupleEqual(orig.shape[1:], (100, 100, 100))
                # check the nearest interpolation mode
                assert_allclose(i.to(torch.uint8).to(torch.float), i.to(torch.float))
                self.assertTupleEqual(i.shape[1:], (101, 100, 107))
        # check labels match
        reverted = i.detach().cpu().numpy().astype(np.int32)
        original = LoadImage(image_only=True)(data[-1])
        n_good = np.sum(np.isclose(reverted, original.numpy(), atol=1e-3))
        reverted_name = i.meta["filename_or_obj"]
        original_name = original.meta["filename_or_obj"]
        self.assertEqual(reverted_name, original_name)
        print("invert diff", reverted.size - n_good)
        self.assertTrue((reverted.size - n_good) < 300000, f"diff. {reverted.size - n_good}")
        set_determinism(seed=None)

    def test_invert_warn_pending(self):
        # this test shouldn't raise a warning or error any more as that issue was fixed
        # by https://github.com/Project-MONAI/MONAI/pull/6257
        set_determinism(seed=0)
        im_fname = make_nifti_image(create_test_image_3d(101, 100, 107, noise_max=100)[1])  # label image, discrete
        transform = Compose(
            [LoadImage(image_only=True), EnsureChannelFirst(), Orientation("RPS"), Lambda(func=lambda x: x)], lazy=True
        )
        output = transform([im_fname for _ in range(2)])
        transform.inverse(output)


if __name__ == "__main__":
    unittest.main()
