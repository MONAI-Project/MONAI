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

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SaveImaged, WriteFileMappingd
from monai.utils import optional_import

nib, has_nib = optional_import("nibabel")


def create_input_file(temp_dir, name):
    test_image = np.random.rand(128, 128, 128)
    input_file = os.path.join(temp_dir, name + ".nii.gz")
    nib.save(nib.Nifti1Image(test_image, np.eye(4)), input_file)
    return input_file


# Test cases that should succeed
SUCCESS_CASES = [(["seg"], ["seg"]), (["image", "seg"], ["seg"])]

# Test cases that should fail
FAILURE_CASES = [(["seg"], ["image"]), (["image"], ["seg"]), (["seg"], ["image", "seg"])]


@unittest.skipUnless(has_nib, "nibabel required")
class TestWriteFileMappingd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir)
        self.mapping_file_path = os.path.join(self.temp_dir, "mapping.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        if os.path.exists(self.mapping_file_path):
            os.remove(self.mapping_file_path)

    def run_test(self, save_keys, write_keys):
        name = "test_image"
        input_file = create_input_file(self.temp_dir, name)
        output_file = os.path.join(self.output_dir, name, name + "_seg.nii.gz")
        data = [{"image": input_file}]

        test_transforms = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])

        post_transforms = Compose(
            [
                SaveImaged(
                    keys=save_keys,
                    meta_keys="image_meta_dict",
                    output_dir=self.output_dir,
                    output_postfix="seg",
                    savepath_in_metadict=True,
                ),
                WriteFileMappingd(keys=write_keys, mapping_file_path=self.mapping_file_path),
            ]
        )

        dataset = Dataset(data=data, transform=test_transforms)
        dataloader = DataLoader(dataset, batch_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32), strides=(2,)).to(device)
        model.eval()

        with torch.no_grad():
            for batch_data in dataloader:
                test_inputs = batch_data["image"].to(device)
                roi_size = (64, 64, 64)
                sw_batch_size = 2
                batch_data["seg"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
                batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

        return input_file, output_file

    @parameterized.expand(SUCCESS_CASES)
    def test_successful_mapping_filed(self, save_keys, write_keys):
        input_file, output_file = self.run_test(save_keys, write_keys)
        self.assertTrue(os.path.exists(self.mapping_file_path))
        with open(self.mapping_file_path) as f:
            mapping_data = json.load(f)
        self.assertEqual(len(mapping_data), len(write_keys))
        for entry in mapping_data:
            self.assertEqual(entry["input"], input_file)
            self.assertEqual(entry["output"], output_file)

    @parameterized.expand(FAILURE_CASES)
    def test_failure_mapping_filed(self, save_keys, write_keys):
        with self.assertRaises(RuntimeError) as cm:
            self.run_test(save_keys, write_keys)

        cause_exception = cm.exception.__cause__
        self.assertIsInstance(cause_exception, KeyError)
        self.assertIn(
            "Missing 'saved_to' key in metadata. Check SaveImage argument 'savepath_in_metadict' is True.",
            str(cause_exception),
        )


if __name__ == "__main__":
    unittest.main()
