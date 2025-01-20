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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import LoadImaged
from monai.transforms.utility.dictionary import SplitDimd
from tests.util import TEST_NDARRAYS, assert_allclose, make_nifti_image, make_rand_affine

TESTS = []
for p in TEST_NDARRAYS:
    for keepdim in (True, False):
        for update_meta in (True, False):
            for list_output in (True, False):
                TESTS.append((keepdim, p, update_meta, list_output))


class TestSplitDimd(unittest.TestCase):
    data: MetaTensor

    @classmethod
    def setUpClass(cls) -> None:
        arr = np.random.rand(2, 10, 8, 7)
        affine = make_rand_affine()
        data = {"i": make_nifti_image(arr, affine)}

        loader = LoadImaged("i", image_only=True)
        cls.data = loader(data)

    @parameterized.expand(TESTS)
    def test_correct(self, keepdim, im_type, update_meta, list_output):
        data = deepcopy(self.data)
        data["i"] = im_type(data["i"])
        arr = data["i"]
        for dim in range(arr.ndim):
            out = SplitDimd("i", dim=dim, keepdim=keepdim, update_meta=update_meta, list_output=list_output)(data)
            if list_output:
                self.assertIsInstance(out, list)
                self.assertEqual(len(out), arr.shape[dim])
            else:
                self.assertIsInstance(out, dict)
                self.assertEqual(len(out.keys()), len(data.keys()) + arr.shape[dim])
            # if updating metadata, pick some random points and
            # check same world coordinates between input and output
            if update_meta:
                for _ in range(10):
                    idx = [np.random.choice(i) for i in arr.shape]
                    split_im_idx = idx[dim]
                    split_idx = deepcopy(idx)
                    split_idx[dim] = 0
                    if list_output:
                        split_im = out[split_im_idx]["i"]
                    else:
                        split_im = out[f"i_{split_im_idx}"]
                    if isinstance(data, MetaTensor) and isinstance(split_im, MetaTensor):
                        # idx[1:] to remove channel and then add 1 for 4th element
                        real_world = data.affine @ torch.tensor(idx[1:] + [1]).double()
                        real_world2 = split_im.affine @ torch.tensor(split_idx[1:] + [1]).double()
                        assert_allclose(real_world, real_world2)

            if list_output:
                out = out[0]["i"]
            else:
                out = out["i_0"]
            expected_ndim = arr.ndim if keepdim else arr.ndim - 1
            self.assertEqual(out.ndim, expected_ndim)
            # assert is a shallow copy
            arr[0, 0, 0, 0] *= 2
            self.assertEqual(arr.flatten()[0], out.flatten()[0])

    def test_singleton(self):
        shape = (2, 1, 8, 7)
        for p in TEST_NDARRAYS:
            arr = p(np.random.rand(*shape))
            out = SplitDimd("i", dim=1)({"i": arr})
            self.assertEqual(out["i"].shape, shape)


if __name__ == "__main__":
    unittest.main()
