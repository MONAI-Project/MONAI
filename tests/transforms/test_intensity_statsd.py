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

import sys
import unittest

import numpy as np
import torch.multiprocessing as mp
from parameterized import parameterized

from monai.data import DataLoader, Dataset
from monai.transforms import IntensityStatsd
from monai.utils.enums import PostFix

TEST_CASE_1 = [
    {"keys": "img", "ops": ["max", "mean"], "key_prefix": "orig", "meta_keys": "test_meta"},
    {"img": np.array([[[0.0, 1.0], [2.0, 3.0]]]), "test_meta": {"affine": None}},
    "test_meta",
    {"orig_max": 3.0, "orig_mean": 1.5},
]

TEST_CASE_2 = [
    {"keys": "img", "ops": "std", "key_prefix": "orig"},
    {"img": np.array([[[0.0, 1.0], [2.0, 3.0]]])},
    PostFix.meta("img"),
    {"orig_std": 1.118034},
]

TEST_CASE_3 = [
    {"keys": "img", "ops": [np.mean, "max", np.min], "key_prefix": "orig"},
    {"img": np.array([[[0.0, 1.0], [2.0, 3.0]]])},
    PostFix.meta("img"),
    {"orig_custom_0": 1.5, "orig_max": 3.0, "orig_custom_1": 0.0},
]

TEST_CASE_4 = [
    {"keys": "img", "ops": ["max", "mean"], "key_prefix": "orig", "channel_wise": True, "meta_key_postfix": "meta"},
    {"img": np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]), "img_meta": {"affine": None}},
    "img_meta",
    {"orig_max": [3.0, 7.0], "orig_mean": [1.5, 5.5]},
]


class TestIntensityStatsd(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, input_param, data, meta_key, expected):
        meta = IntensityStatsd(**input_param)(data)[meta_key]
        for k, v in expected.items():
            self.assertTrue(k in meta)
            np.testing.assert_allclose(v, meta[k], atol=1e-3)

    def test_dataloader(self):
        dataset = Dataset(
            data=[{"img": np.array([[[0.0, 1.0], [2.0, 3.0]]])}, {"img": np.array([[[0.0, 1.0], [2.0, 3.0]]])}],
            transform=IntensityStatsd(keys="img", ops=["max", "mean"], key_prefix="orig"),
        )
        # set num workers = 0 for mac / win
        num_workers = 2 if sys.platform == "linux" else 0
        dataloader = DataLoader(dataset=dataset, num_workers=num_workers, batch_size=2)
        orig_method = mp.get_start_method()
        mp.set_start_method("spawn", force=True)

        for d in dataloader:
            meta = d[PostFix.meta("img")]
            np.testing.assert_allclose(meta["orig_max"], [3.0, 3.0], atol=1e-3)
            np.testing.assert_allclose(meta["orig_mean"], [1.5, 1.5], atol=1e-3)
        # restore the mp method
        mp.set_start_method(orig_method, force=True)

    def test_mask(self):
        data = {"img": np.array([[[0.0, 1.0], [2.0, 3.0]]]), "img_mask": np.array([[[1, 0], [1, 0]]], dtype=bool)}
        stats = IntensityStatsd(keys="img", ops=["max", "mean"], mask_keys="img_mask", key_prefix="orig")
        meta = stats(data)[PostFix.meta("img")]
        np.testing.assert_allclose(meta["orig_max"], 2.0, atol=1e-3)
        np.testing.assert_allclose(meta["orig_mean"], 1.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
