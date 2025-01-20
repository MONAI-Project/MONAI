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
from pathlib import Path

from monai.data import load_decathlon_datalist


class TestLoadDecathlonDatalist(unittest.TestCase):

    def test_seg_values(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_data = {
                "name": "Spleen",
                "description": "Spleen Segmentation",
                "labels": {"0": "background", "1": "spleen"},
                "training": [
                    {"image": "spleen_19.nii.gz", "label": "spleen_19.nii.gz"},
                    {"image": "spleen_31.nii.gz", "label": "spleen_31.nii.gz"},
                ],
                "test": [{"image": "spleen_15.nii.gz"}, {"image": "spleen_23.nii.gz"}],
            }
            json_str = json.dumps(test_data)
            file_path = os.path.join(tempdir, "test_data.json")
            with open(file_path, "w") as json_file:
                json_file.write(json_str)
            result = load_decathlon_datalist(file_path, True, "training", tempdir)
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "spleen_19.nii.gz"))
            self.assertEqual(result[0]["label"], os.path.join(tempdir, "spleen_19.nii.gz"))
            result = load_decathlon_datalist(file_path, True, "test", None)
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "spleen_15.nii.gz"))

    def test_cls_values(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_data = {
                "name": "ChestXRay",
                "description": "Chest X-ray classification",
                "labels": {"0": "background", "1": "chest"},
                "training": [{"image": "chest_19.nii.gz", "label": 0}, {"image": "chest_31.nii.gz", "label": 1}],
                "test": ["chest_15.nii.gz", "chest_23.nii.gz"],
            }
            json_str = json.dumps(test_data)
            file_path = os.path.join(tempdir, "test_data.json")
            with open(file_path, "w") as json_file:
                json_file.write(json_str)
            result = load_decathlon_datalist(file_path, False, "training", tempdir)
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "chest_19.nii.gz"))
            self.assertEqual(result[0]["label"], 0)

    def test_seg_no_basedir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_data = {
                "name": "Spleen",
                "description": "Spleen Segmentation",
                "labels": {"0": "background", "1": "spleen"},
                "training": [
                    {
                        "image": os.path.join(tempdir, "spleen_19.nii.gz"),
                        "label": os.path.join(tempdir, "spleen_19.nii.gz"),
                    },
                    {
                        "image": os.path.join(tempdir, "spleen_31.nii.gz"),
                        "label": os.path.join(tempdir, "spleen_31.nii.gz"),
                    },
                ],
                "test": [os.path.join(tempdir, "spleen_15.nii.gz"), os.path.join(tempdir, "spleen_23.nii.gz")],
            }
            json_str = json.dumps(test_data)
            file_path = os.path.join(tempdir, "test_data.json")
            with open(file_path, "w") as json_file:
                json_file.write(json_str)
            result = load_decathlon_datalist(file_path, True, "training", None)
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "spleen_19.nii.gz"))
            self.assertEqual(result[0]["label"], os.path.join(tempdir, "spleen_19.nii.gz"))
            result = load_decathlon_datalist(file_path, True, "test", None)
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "spleen_15.nii.gz"))

    def test_seg_no_labels(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_data = {
                "name": "Spleen",
                "description": "Spleen Segmentation",
                "labels": {"0": "background", "1": "spleen"},
                "test": ["spleen_15.nii.gz", "spleen_23.nii.gz"],
            }
            json_str = json.dumps(test_data)
            file_path = os.path.join(tempdir, "test_data.json")
            with open(file_path, "w") as json_file:
                json_file.write(json_str)
            result = load_decathlon_datalist(file_path, True, "test", tempdir)
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "spleen_15.nii.gz"))

    def test_additional_items(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "mask31.txt"), "w") as f:
                f.write("spleen31 mask")

            test_data = {
                "name": "Spleen",
                "description": "Spleen Segmentation",
                "labels": {"0": "background", "1": "spleen"},
                "training": [
                    {"image": "spleen_19.nii.gz", "label": "spleen_19.nii.gz", "mask": "spleen mask"},
                    {"image": "spleen_31.nii.gz", "label": "spleen_31.nii.gz", "mask": "mask31.txt"},
                ],
                "test": ["spleen_15.nii.gz", "spleen_23.nii.gz"],
            }
            json_str = json.dumps(test_data)
            file_path = os.path.join(tempdir, "test_data.json")
            with open(file_path, "w") as json_file:
                json_file.write(json_str)
            result = load_decathlon_datalist(file_path, True, "training", Path(tempdir))
            self.assertEqual(result[0]["image"], os.path.join(tempdir, "spleen_19.nii.gz"))
            self.assertEqual(result[0]["label"], os.path.join(tempdir, "spleen_19.nii.gz"))
            self.assertEqual(result[1]["mask"], os.path.join(tempdir, "mask31.txt"))
            self.assertEqual(result[0]["mask"], "spleen mask")


if __name__ == "__main__":
    unittest.main()
