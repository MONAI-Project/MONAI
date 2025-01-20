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

import tempfile
import unittest
from io import BytesIO

import numpy as np

from monai.data import NPZDictItemDataset


class TestNPZDictItemDataset(unittest.TestCase):

    def test_load_stream(self):
        dat0 = np.random.rand(10, 1, 4, 4)
        dat1 = np.random.rand(10, 1, 4, 4)

        npzfile = BytesIO()
        np.savez_compressed(npzfile, dat0=dat0, dat1=dat1)
        npzfile.seek(0)

        npzds = NPZDictItemDataset(npzfile, {"dat0": "images", "dat1": "seg"})

        item = npzds[0]

        np.testing.assert_allclose(item["images"].shape, (1, 4, 4))
        np.testing.assert_allclose(item["seg"].shape, (1, 4, 4))

    def test_load_file(self):
        dat0 = np.random.rand(10, 1, 4, 4)
        dat1 = np.random.rand(10, 1, 4, 4)

        with tempfile.TemporaryDirectory() as tempdir:
            npzfile = f"{tempdir}/test.npz"

            np.savez_compressed(npzfile, dat0=dat0, dat1=dat1)

            npzds = NPZDictItemDataset(npzfile, {"dat0": "images", "dat1": "seg"})

            item = npzds[0]

            np.testing.assert_allclose(item["images"].shape, (1, 4, 4))
            np.testing.assert_allclose(item["seg"].shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
