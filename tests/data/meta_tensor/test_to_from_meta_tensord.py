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

import random
import string
import unittest
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai import config
from monai.data.meta_tensor import MetaTensor
from monai.transforms import FromMetaTensord, ToMetaTensord
from monai.utils.enums import PostFix
from tests.util import TEST_DEVICES, assert_allclose

DTYPES = [[torch.float32], [torch.float64], [torch.float16], [torch.int64], [torch.int32]]
TESTS = []
for _device in TEST_DEVICES:
    for _dtype in DTYPES:
        for _data_type in ("tensor", "numpy"):
            TESTS.append((*_device, *_dtype, _data_type))


def rand_string(min_len=5, max_len=10):
    str_size = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.punctuation
    return "".join(random.choice(chars) for _ in range(str_size))


@unittest.skipIf(config.USE_META_DICT, "skipping not metatensor")
class TestToFromMetaTensord(unittest.TestCase):
    @staticmethod
    def get_im(shape=None, dtype=None, device=None):
        if shape is None:
            shape = (1, 10, 8)
        affine = torch.randint(0, 10, (4, 4))
        meta = {"fname": rand_string()}
        t = torch.rand(shape)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        m = MetaTensor(t.clone(), affine, meta)
        return m

    def check_ids(self, a, b, should_match):
        comp = self.assertEqual if should_match else self.assertNotEqual
        comp(id(a), id(b))

    def check(
        self,
        out: torch.Tensor,
        orig: torch.Tensor,
        *,
        shape: bool = True,
        vals: bool = True,
        ids: bool = True,
        device: str | torch.device | None = None,
        meta: bool = True,
        check_ids: bool = False,
        **kwargs,
    ):
        if device is None:
            device = orig.device

        # check the image
        self.assertIsInstance(out, type(orig))
        if shape:
            assert_allclose(torch.as_tensor(out.shape), torch.as_tensor(orig.shape))
        if vals:
            assert_allclose(out, orig, **kwargs)
        if check_ids:
            self.check_ids(out, orig, ids)
        self.assertTrue(str(device) in str(out.device))

        # check meta and affine are equal and affine is on correct device
        if isinstance(orig, MetaTensor) and isinstance(out, MetaTensor) and meta:
            orig_meta_no_affine = deepcopy(orig.meta)
            del orig_meta_no_affine["affine"]
            out_meta_no_affine = deepcopy(out.meta)
            del out_meta_no_affine["affine"]
            self.assertEqual(orig_meta_no_affine, out_meta_no_affine)
            assert_allclose(out.affine, orig.affine)
            if check_ids:
                self.check_ids(out.affine, orig.affine, ids)
                self.check_ids(out.meta, orig.meta, ids)

    @parameterized.expand(TESTS)
    def test_from_to_meta_tensord(self, device, dtype, data_type="tensor"):
        m1 = self.get_im(device=device, dtype=dtype)
        m2 = self.get_im(device=device, dtype=dtype)
        m3 = self.get_im(device=device, dtype=dtype)
        d_metas = {"m1": m1, "m2": m2, "m3": m3}
        m1_meta = {k: v for k, v in m1.meta.items() if k != "affine"}
        m1_aff = m1.affine

        # FROM -> forward
        t_from_meta = FromMetaTensord(["m1", "m2"], data_type=data_type)
        d_dict = t_from_meta(d_metas)

        self.assertEqual(
            sorted(d_dict.keys()),
            [
                "m1",
                PostFix.meta("m1"),
                PostFix.transforms("m1"),
                "m2",
                PostFix.meta("m2"),
                PostFix.transforms("m2"),
                "m3",
            ],
        )
        self.check(d_dict["m3"], m3, ids=True)  # unchanged
        if data_type == "tensor":
            self.check(d_dict["m1"], m1.as_tensor(), ids=False)
        else:
            self.assertIsInstance(d_dict["m1"], np.ndarray)
        meta_out = {k: v for k, v in d_dict["m1_meta_dict"].items() if k != "affine"}
        aff_out = d_dict["m1_meta_dict"]["affine"]
        self.check(aff_out, m1_aff, ids=False)
        self.assertEqual(meta_out, m1_meta)

        # FROM -> inverse
        d_meta_dict_meta = t_from_meta.inverse(d_dict)
        self.assertEqual(sorted(d_meta_dict_meta.keys()), ["m1", "m2", "m3"])
        if data_type == "numpy":
            m1, m1_aff = m1.cpu(), m1_aff.cpu()
        self.check(d_meta_dict_meta["m1"], m1, ids=False)
        meta_out = {k: v for k, v in d_meta_dict_meta["m1"].meta.items() if k != "affine"}
        aff_out = d_meta_dict_meta["m1"].affine
        self.check(aff_out, m1_aff, ids=False)
        self.assertEqual(meta_out, m1_meta)

        # TO -> Forward
        t_to_meta = ToMetaTensord(["m1", "m2"])
        d_dict_meta = t_to_meta(d_dict)
        self.assertEqual(sorted(d_dict_meta.keys()), ["m1", "m2", "m3"], f"flag: {config.USE_META_DICT}")
        self.check(d_dict_meta["m3"], m3, ids=True)  # unchanged (except deep copy in inverse)
        self.check(d_dict_meta["m1"], m1, ids=False)
        meta_out = {k: v for k, v in d_dict_meta["m1"].meta.items() if k != "affine"}
        aff_out = d_dict_meta["m1"].meta["affine"]
        self.check(aff_out, m1_aff, ids=False)
        self.assertEqual(meta_out, m1_meta)

        # TO -> Inverse
        d_dict_meta_dict = t_to_meta.inverse(d_dict_meta)
        self.assertEqual(
            sorted(d_dict_meta_dict.keys()),
            [
                "m1",
                PostFix.meta("m1"),
                PostFix.transforms("m1"),
                "m2",
                PostFix.meta("m2"),
                PostFix.transforms("m2"),
                "m3",
            ],
        )
        self.check(d_dict_meta_dict["m3"], m3.as_tensor(), ids=False)  # unchanged (except deep copy in inverse)
        self.check(d_dict_meta_dict["m1"], m1.as_tensor(), ids=False)
        meta_out = {k: v for k, v in d_dict_meta_dict["m1_meta_dict"].items() if k != "affine"}
        aff_out = d_dict_meta_dict["m1_meta_dict"]["affine"]
        self.check(aff_out, m1_aff, ids=False)
        self.assertEqual(meta_out, m1_meta)


if __name__ == "__main__":
    unittest.main()
