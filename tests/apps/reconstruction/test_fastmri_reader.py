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

import numpy as np
from parameterized import parameterized

from monai.apps.reconstruction.fastmri_reader import FastMRIReader
from tests.util import SkipIfNoModule, assert_allclose

TEST_CASE1 = [
    {
        "kspace": np.array([[1.0, 2.0]]),
        "filename": "test1",
        "reconstruction_rss": np.array([[1.0, 2.0]]),
        "acquisition": "FS",
        "max": 2.0,
        "norm": 2.2,
        "patient_id": 12,
    },
    np.array([[1.0, 2.0]]),
    {
        "filename": "test1",
        "reconstruction_rss": np.array([[1.0, 2.0]]),
        "acquisition": "FS",
        "max": 2.0,
        "norm": 2.2,
        "patient_id": 12,
        "mask": np.zeros([1, 2]),
    },
]

TEST_CASE2 = [
    {
        "kspace": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "filename": "test2",
        "reconstruction_rss": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "acquisition": "FS",
        "max": 4.0,
        "norm": 5.5,
        "patient_id": 1234,
    },
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    {
        "filename": "test2",
        "reconstruction_rss": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "acquisition": "FS",
        "max": 4.0,
        "norm": 5.5,
        "patient_id": 1234,
        "mask": np.zeros([2, 2]),
    },
]


@SkipIfNoModule("h5py")
class TestMRIUtils(unittest.TestCase):
    @parameterized.expand([TEST_CASE1, TEST_CASE2])
    def test_get_data(self, test_data, test_res, test_meta):
        reader = FastMRIReader()
        res, meta = reader.get_data(test_data)
        assert_allclose(res, test_res)
        for key in test_meta:
            if isinstance(test_meta[key], np.ndarray):
                assert_allclose(test_meta[key], meta[key])
            else:
                self.assertEqual(test_meta[key], meta[key])


if __name__ == "__main__":
    unittest.main()
