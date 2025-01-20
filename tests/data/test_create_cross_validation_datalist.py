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
import tempfile
import unittest
from pathlib import Path

from monai.data import create_cross_validation_datalist, load_decathlon_datalist


class TestCreateCrossValidationDatalist(unittest.TestCase):

    def test_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            datalist = []
            for i in range(5):
                image = os.path.join(tempdir, f"test_image{i}.nii.gz")
                label = os.path.join(tempdir, f"test_label{i}.nii.gz")
                Path(image).touch()
                Path(label).touch()
                datalist.append({"image": image, "label": label})

            filename = os.path.join(tempdir, "test_datalist.json")
            result = create_cross_validation_datalist(
                datalist=datalist,
                nfolds=5,
                train_folds=[0, 1, 2, 3],
                val_folds=4,
                train_key="test_train",
                val_key="test_val",
                filename=Path(filename),
                shuffle=True,
                seed=123,
                check_missing=True,
                keys=["image", "label"],
                root_dir=None,
                allow_missing_keys=False,
                raise_error=True,
            )

            loaded = load_decathlon_datalist(filename, data_list_key="test_train")
            for r, l in zip(result["test_train"], loaded):
                self.assertEqual(r["image"], l["image"])
                self.assertEqual(r["label"], l["label"])


if __name__ == "__main__":
    unittest.main()
