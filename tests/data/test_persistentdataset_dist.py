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
import shutil
import tempfile
import unittest

import numpy as np
import torch.distributed as dist

from monai.data import PersistentDataset, json_hashing
from monai.transforms import Transform
from tests.util import DistCall, DistTestCase


class _InplaceXform(Transform):
    def __call__(self, data):
        if data:
            data[0] = data[0] + np.pi
        else:
            data.append(1)
        return data


class TestDistDataset(DistTestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @DistCall(nnodes=1, nproc_per_node=2)
    def test_mp_dataset(self):
        print("persistent", dist.get_rank())
        items = [[list(range(i))] for i in range(5)]
        ds = PersistentDataset(items, transform=_InplaceXform(), cache_dir=self.tempdir)
        self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
        ds1 = PersistentDataset(items, transform=_InplaceXform(), cache_dir=self.tempdir)
        self.assertEqual(list(ds1), list(ds))
        self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])

        ds = PersistentDataset(items, transform=_InplaceXform(), cache_dir=self.tempdir, hash_func=json_hashing)
        self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
        ds1 = PersistentDataset(items, transform=_InplaceXform(), cache_dir=self.tempdir, hash_func=json_hashing)
        self.assertEqual(list(ds1), list(ds))
        self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])


class TestDistCreateDataset(DistTestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    @DistCall(nnodes=1, nproc_per_node=2)
    def test_mp_dataset(self):
        print("persistent", dist.get_rank())
        items = [[list(range(i))] for i in range(5)]
        cache_dir = os.path.join(self.tempdir, "test")
        ds = PersistentDataset(items, transform=_InplaceXform(), cache_dir=cache_dir)
        self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
        ds1 = PersistentDataset(items, transform=_InplaceXform(), cache_dir=cache_dir)
        self.assertEqual(list(ds1), list(ds))
        self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])


if __name__ == "__main__":
    unittest.main()
