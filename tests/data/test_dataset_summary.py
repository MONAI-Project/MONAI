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

import glob
import os
import tempfile
import unittest

import nibabel as nib
import numpy as np

from monai.data import Dataset, DatasetSummary, create_test_image_3d
from monai.transforms import LoadImaged
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToNumpyd
from monai.utils import set_determinism


def test_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, np.ndarray):
        return np.stack(batch, 0)
    elif isinstance(elem, dict):
        return elem_type({key: test_collate([d[key] for d in batch]) for key in elem})


class TestDatasetSummary(unittest.TestCase):

    def test_spacing_intensity(self):
        set_determinism(seed=0)
        with tempfile.TemporaryDirectory() as tempdir:
            for i in range(5):
                im, seg = create_test_image_3d(32, 32, 32, num_seg_classes=1, num_objs=3, rad_max=6, channel_dim=0)
                n = nib.Nifti1Image(im, np.eye(4))
                nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))
                n = nib.Nifti1Image(seg, np.eye(4))
                nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

            train_images = sorted(glob.glob(os.path.join(tempdir, "img*.nii.gz")))
            train_labels = sorted(glob.glob(os.path.join(tempdir, "seg*.nii.gz")))
            data_dicts = [
                {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
            ]

            t = Compose(
                [
                    LoadImaged(keys=["image", "label"], image_only=False),
                    ToNumpyd(keys=["image", "label", "image_meta_dict", "label_meta_dict"]),
                ]
            )
            dataset = Dataset(data=data_dicts, transform=t)

            # test **kwargs of `DatasetSummary` for `DataLoader`
            calculator = DatasetSummary(dataset, num_workers=4, meta_key="image_meta_dict", collate_fn=test_collate)

            target_spacing = calculator.get_target_spacing(spacing_key="pixdim")
            self.assertEqual(target_spacing, (1.0, 1.0, 1.0))
            calculator.calculate_statistics()
            np.testing.assert_allclose(calculator.data_mean, 0.892599, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(calculator.data_std, 0.131731, rtol=1e-5, atol=1e-5)
            calculator.calculate_percentiles(sampling_flag=True, interval=2)
            self.assertEqual(calculator.data_max_percentile, 1.0)
            np.testing.assert_allclose(calculator.data_min_percentile, 0.556411, rtol=1e-5, atol=1e-5)

    def test_anisotropic_spacing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            pixdims = [[1.0, 1.0, 5.0], [1.0, 1.0, 4.0], [1.0, 1.0, 4.5], [1.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
            for i in range(5):
                im, seg = create_test_image_3d(32, 32, 32, num_seg_classes=1, num_objs=3, rad_max=6, channel_dim=0)
                n = nib.Nifti1Image(im, np.eye(4))
                n.header["pixdim"][1:4] = pixdims[i]
                nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))
                n = nib.Nifti1Image(seg, np.eye(4))
                n.header["pixdim"][1:4] = pixdims[i]
                nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

            train_images = sorted(glob.glob(os.path.join(tempdir, "img*.nii.gz")))
            train_labels = sorted(glob.glob(os.path.join(tempdir, "seg*.nii.gz")))
            data_dicts = [
                {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
            ]

            t = Compose([LoadImaged(keys=["image", "label"])])
            dataset = Dataset(data=data_dicts, transform=t)

            calculator = DatasetSummary(dataset, num_workers=4)

            target_spacing = calculator.get_target_spacing(anisotropic_threshold=4.0, percentile=20.0)
            np.testing.assert_allclose(target_spacing, (1.0, 1.0, 1.8))


if __name__ == "__main__":
    unittest.main()
