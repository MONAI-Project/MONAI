# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import unittest
import numpy as np

from monai.data import DataLoader
from monai.data import CacheDataset, create_test_image_2d
from monai.transforms import AddChanneld, Compose, LoadImaged, ToTensord, SpatialPadd, RandFlipd
from monai.data.utils import decollate_batch
from monai.utils import set_determinism
from tests.utils import make_nifti_image

from parameterized import parameterized


set_determinism(seed=0)

IM_2D_FNAME = make_nifti_image(create_test_image_2d(100, 101)[0])

DATA_2D = {"image": IM_2D_FNAME}

TESTS = []
TESTS.append((
    "2D",
    [DATA_2D for _ in range(6)],
))

class TestDeCollate(unittest.TestCase):
    def check_match(self, in1, in2):
        if isinstance(in1, dict):
            self.assertTrue(isinstance(in2, dict))
            self.check_match(list(in1.keys()), list(in2.keys()))
            self.check_match(list(in1.values()), list(in2.values()))
        elif any(isinstance(in1, i) for i in [list, tuple]):
            for l1, l2 in zip(in1, in2):
                self.check_match(l1, l2)
        elif any(isinstance(in1, i) for i in [str, int]):
            self.assertEqual(in1, in2)
        elif any(isinstance(in1, i) for i in [torch.Tensor, np.ndarray]):
            np.testing.assert_array_equal(in1, in2)
        else:
            raise RuntimeError(f"Not sure how to compare types. type(in1): {type(in1)}, type(in2): {type(in2)}")

    @parameterized.expand(TESTS)
    def test_decollation(self, _, data, batch_size=2, num_workers=0):
        transforms = Compose([
            LoadImaged("image"),
            AddChanneld("image"),
            SpatialPadd("image", 150),
            RandFlipd("image", prob=1., spatial_axis=1),
            ToTensord("image"),
        ])
        dataset = CacheDataset(data, transforms, progress=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for b, batch_data in enumerate(loader):
            decollated = decollate_batch(batch_data)

            for i, d in enumerate(decollated):
                self.check_match(dataset[b * batch_size + i], d)


if __name__ == "__main__":
    unittest.main()
