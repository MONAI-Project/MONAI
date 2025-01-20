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

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch.nn as nn

from monai.data import DataLoader, Dataset, IterableDataset
from monai.engines import SupervisedEvaluator
from monai.transforms import Compose, LoadImaged, SimulateDelayd


class _Stream:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class TestIterableDataset(unittest.TestCase):

    def test_shape(self):
        expected_shape = (128, 128, 128)
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]).astype(float), np.eye(4))
        test_data = []
        with tempfile.TemporaryDirectory() as tempdir:
            for i in range(6):
                nib.save(test_image, os.path.join(tempdir, f"test_image{str(i)}.nii.gz"))
                test_data.append({"image": os.path.join(tempdir, f"test_image{i}.nii.gz")})

            test_transform = Compose([LoadImaged(keys="image"), SimulateDelayd(keys="image", delay_time=1e-7)])

            data_iterator = _Stream(test_data)
            with self.assertRaises(TypeError):  # Dataset doesn't work
                dataset = Dataset(data=data_iterator, transform=test_transform)
                for _ in dataset:
                    pass
            dataset = IterableDataset(data=data_iterator, transform=test_transform)
            for d in dataset:
                self.assertTupleEqual(d["image"].shape, expected_shape)

            num_workers = 2 if sys.platform == "linux" else 0
            dataloader = DataLoader(dataset=dataset, batch_size=3, num_workers=num_workers)
            for d in dataloader:
                self.assertTupleEqual(d["image"].shape[1:], expected_shape)

    def test_supervisedevaluator(self):
        """
        Test that a SupervisedEvaluator is compatible with IterableDataset in conjunction with DataLoader.
        """
        data = list(range(10))
        dl = DataLoader(IterableDataset(data))
        evaluator = SupervisedEvaluator(device="cpu", val_data_loader=dl, network=nn.Identity())
        evaluator.run()  # fails if the epoch length or other internal setup is not done correctly

        self.assertEqual(evaluator.state.iteration, len(data))


if __name__ == "__main__":
    unittest.main()
