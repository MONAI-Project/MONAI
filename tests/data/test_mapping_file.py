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
from parameterized import parameterized

from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImage, SaveImage, WriteFileMapping
from monai.utils import optional_import

nib, has_nib = optional_import("nibabel")


def create_input_file(temp_dir, name):
    test_image = np.random.rand(128, 128, 128)
    output_ext = ".nii.gz"
    input_file = os.path.join(temp_dir, name + output_ext)
    nib.save(nib.Nifti1Image(test_image, np.eye(4)), input_file)
    return input_file


def create_transform(temp_dir, mapping_file_path, savepath_in_metadict=True):
    return Compose(
        [
            LoadImage(image_only=True),
            SaveImage(output_dir=temp_dir, output_ext=".nii.gz", savepath_in_metadict=savepath_in_metadict),
            WriteFileMapping(mapping_file_path=mapping_file_path),
        ]
    )


@unittest.skipUnless(has_nib, "nibabel required")
class TestWriteFileMapping(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @parameterized.expand([(True,), (False,)])
    def test_mapping_file(self, savepath_in_metadict):
        mapping_file_path = os.path.join(self.temp_dir, "mapping.json")
        name = "test_image"
        input_file = create_input_file(self.temp_dir, name)
        output_file = os.path.join(self.temp_dir, name, name + "_trans.nii.gz")

        transform = create_transform(self.temp_dir, mapping_file_path, savepath_in_metadict)

        if savepath_in_metadict:
            transform(input_file)
            self.assertTrue(os.path.exists(mapping_file_path))
            with open(mapping_file_path) as f:
                mapping_data = json.load(f)
            self.assertEqual(len(mapping_data), 1)
            self.assertEqual(mapping_data[0]["input"], input_file)
            self.assertEqual(mapping_data[0]["output"], output_file)
        else:
            with self.assertRaises(RuntimeError) as cm:
                transform(input_file)
            cause_exception = cm.exception.__cause__
            self.assertIsInstance(cause_exception, KeyError)
            self.assertIn(
                "Missing 'saved_to' key in metadata. Check SaveImage argument 'savepath_in_metadict' is True.",
                str(cause_exception),
            )

    def test_multiprocess_mapping_file(self):
        num_images = 50

        single_mapping_file = os.path.join(self.temp_dir, "single_mapping.json")
        multi_mapping_file = os.path.join(self.temp_dir, "multi_mapping.json")

        data = [create_input_file(self.temp_dir, f"test_image_{i}") for i in range(num_images)]

        # single process
        single_transform = create_transform(self.temp_dir, single_mapping_file)
        single_dataset = Dataset(data=data, transform=single_transform)
        single_loader = DataLoader(single_dataset, batch_size=1, num_workers=0, shuffle=True)
        for _ in single_loader:
            pass

        # multiple processes
        multi_transform = create_transform(self.temp_dir, multi_mapping_file)
        multi_dataset = Dataset(data=data, transform=multi_transform)
        multi_loader = DataLoader(multi_dataset, batch_size=4, num_workers=3, shuffle=True)
        for _ in multi_loader:
            pass

        with open(single_mapping_file) as f:
            single_mapping_data = json.load(f)
        with open(multi_mapping_file) as f:
            multi_mapping_data = json.load(f)

        single_set = {(entry["input"], entry["output"]) for entry in single_mapping_data}
        multi_set = {(entry["input"], entry["output"]) for entry in multi_mapping_data}

        self.assertEqual(single_set, multi_set)


if __name__ == "__main__":
    unittest.main()
