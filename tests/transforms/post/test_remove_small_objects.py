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
import torch
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms.post.array import RemoveSmallObjects
from monai.transforms.post.dictionary import RemoveSmallObjectsd
from monai.utils import optional_import
from tests.util import TEST_NDARRAYS, SkipIfNoModule, assert_allclose

morphology, has_morphology = optional_import("skimage.morphology")

TEST_ZEROS = np.zeros((1, 9, 8, 7))
TEST_ONES = np.ones((3, 7, 8, 9))

TEST_INPUT1 = np.array([[[0, 0, 2, 1, 0], [1, 1, 1, 2, 0], [1, 1, 1, 0, 1]]])

TEST_OUTPUT1 = np.array([[[0, 0, 2, 1, 0], [1, 1, 1, 2, 0], [1, 1, 1, 0, 0]]])

TEST_INPUT2 = np.array([[[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [0, 0, 0, 1, 1]]])
affine = torch.eye(4, dtype=torch.float64)
affine[0, 0] = 2.0
TEST_INPUT3 = MetaTensor(TEST_INPUT2, affine=affine)

TESTS: list[tuple] = []
for dtype in (int, float):
    for p in TEST_NDARRAYS:
        TESTS.append((dtype, p, TEST_ZEROS, None))
        TESTS.append((dtype, p, TEST_ONES, None))
        TESTS.append((dtype, p, TEST_INPUT1, None, {"min_size": 6}))
        TESTS.append((dtype, p, TEST_INPUT1, None, {"min_size": 7, "connectivity": 2}))
        # for non-independent channels, the twos should stay
        TESTS.append((dtype, p, TEST_INPUT1, TEST_OUTPUT1, {"min_size": 2, "independent_channels": False}))

TESTS_PHYSICAL: list[tuple] = []
for dtype in (int, float):
    TESTS_PHYSICAL.append((dtype, np.array, TEST_INPUT2, None, {"min_size": 3, "by_measure": True, "pixdim": (2, 1)}))
    TESTS_PHYSICAL.append((dtype, MetaTensor, TEST_INPUT3, None, {"min_size": 3, "by_measure": True}))


@SkipIfNoModule("skimage.morphology")
class TestRemoveSmallObjects(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_remove_small_objects(self, dtype, im_type, lbl, expected, params=None):
        params = params or {}
        if expected is None:
            dtype = bool if len(np.unique(lbl)) == 1 else int
            expected = morphology.remove_small_objects(lbl.astype(dtype), **params)
        expected = im_type(expected, dtype=dtype)
        lbl = im_type(lbl, dtype=dtype)
        lbl_clean = RemoveSmallObjects(**params)(lbl)
        assert_allclose(lbl_clean, expected, device_test=True)
        if isinstance(lbl, MetaTensor):
            assert_allclose(lbl.affine, lbl_clean.affine)

    @parameterized.expand(TESTS_PHYSICAL)
    def test_remove_small_objects_physical(self, dtype, im_type, lbl, expected, params):
        params = params or {}
        min_size = np.ceil(params["min_size"] / 2)

        if expected is None:
            dtype = bool if lbl.max() <= 1 else int
            expected = morphology.remove_small_objects(lbl.astype(dtype), min_size=min_size)
        expected = im_type(expected, dtype=dtype)
        lbl = im_type(lbl, dtype=dtype)
        lbl_clean = RemoveSmallObjects(**params)(lbl)
        assert_allclose(lbl_clean, expected, device_test=True)

        lbl_clean = RemoveSmallObjectsd("lbl", **params)({"lbl": lbl})["lbl"]
        assert_allclose(lbl_clean, expected, device_test=True)

    @parameterized.expand(TESTS)
    def test_remove_small_objects_dict(self, dtype, im_type, lbl, expected, params=None):
        params = params or {}
        if expected is None:
            dtype = bool if len(np.unique(lbl)) == 1 else int
            expected = morphology.remove_small_objects(lbl.astype(dtype), **params)
        expected = im_type(expected, dtype=dtype)
        lbl = im_type(lbl, dtype=dtype)
        lbl_clean = RemoveSmallObjectsd("lbl", **params)({"lbl": lbl})["lbl"]
        assert_allclose(lbl_clean, expected, device_test=True)
        if isinstance(lbl, MetaTensor):
            assert_allclose(lbl.affine, lbl_clean.affine)


if __name__ == "__main__":
    unittest.main()
