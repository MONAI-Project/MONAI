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
import tempfile
import unittest

from monai.data import Dataset, DatasetFunc, load_decathlon_datalist, partition_dataset


class TestDatasetFunc(unittest.TestCase):

    def test_seg_values(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # prepare test datalist file
            test_data = {
                "name": "Spleen",
                "description": "Spleen Segmentation",
                "labels": {"0": "background", "1": "spleen"},
                "training": [
                    {"image": "spleen_19.nii.gz", "label": "spleen_19.nii.gz"},
                    {"image": "spleen_31.nii.gz", "label": "spleen_31.nii.gz"},
                ],
                "test": ["spleen_15.nii.gz", "spleen_23.nii.gz"],
            }
            json_str = json.dumps(test_data)
            file_path = os.path.join(tempdir, "test_data.json")
            with open(file_path, "w") as json_file:
                json_file.write(json_str)

            data_list = DatasetFunc(
                data=file_path, func=load_decathlon_datalist, data_list_key="training", base_dir=tempdir
            )
            # partition dataset for train / validation
            data_partition = DatasetFunc(
                data=data_list, func=lambda x, **kwargs: partition_dataset(x, **kwargs)[0], num_partitions=2
            )
            dataset = Dataset(data=data_partition, transform=None)
            self.assertEqual(dataset[0]["image"], os.path.join(tempdir, "spleen_19.nii.gz"))
            self.assertEqual(dataset[0]["label"], os.path.join(tempdir, "spleen_19.nii.gz"))


if __name__ == "__main__":
    unittest.main()
