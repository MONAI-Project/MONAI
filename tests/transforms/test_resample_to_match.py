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

import itertools
import os
import random
import shutil
import string
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.data.image_reader import ITKReader, NibabelReader
from monai.data.image_writer import ITKWriter
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ResampleToMatch, SaveImage, SaveImaged
from monai.utils import optional_import
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import assert_allclose, download_url_or_skip_test, testing_data_config

_, has_itk = optional_import("itk", allow_namespace_pkg=True)

TEST_CASES = ["itkreader", "nibabelreader"]


def get_rand_fname(len=10, suffix=".nii.gz"):
    letters = string.ascii_letters
    out = "".join(random.choice(letters) for _ in range(len))
    out += suffix
    return out


@unittest.skipUnless(has_itk, "itk not installed")
class TestResampleToMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(__class__, cls).setUpClass()
        cls.fnames = []
        cls.tmpdir = tempfile.mkdtemp()
        for key in ("0000_t2_tse_tra_4", "0000_ep2d_diff_tra_7"):
            fname = os.path.join(cls.tmpdir, f"test_{key}.nii.gz")
            url = testing_data_config("images", key, "url")
            hash_type = testing_data_config("images", key, "hash_type")
            hash_val = testing_data_config("images", key, "hash_val")
            download_url_or_skip_test(url=url, filepath=fname, hash_type=hash_type, hash_val=hash_val)
            cls.fnames.append(fname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        super(__class__, cls).tearDownClass()

    @parameterized.expand(itertools.product([NibabelReader, ITKReader], ["monai.data.NibabelWriter", ITKWriter]))
    def test_correct(self, reader, writer):
        loader = Compose([LoadImaged(("im1", "im2"), reader=reader), EnsureChannelFirstd(("im1", "im2"))])
        data = loader({"im1": self.fnames[0], "im2": self.fnames[1]})
        tr = ResampleToMatch()
        im_mod = tr(data["im2"], data["im1"])

        # check lazy resample
        call_param = {"img": data["im2"], "img_dst": data["im1"]}
        test_resampler_lazy(tr, im_mod, init_param={}, call_param=call_param)

        saver = SaveImaged(
            "im3", output_dir=self.tmpdir, output_postfix="", separate_folder=False, writer=writer, resample=False
        )
        im_mod.meta["filename_or_obj"] = get_rand_fname()
        saver({"im3": im_mod})

        saved = nib.load(os.path.join(self.tmpdir, im_mod.meta["filename_or_obj"]))
        assert_allclose(data["im1"].shape[1:], saved.shape)
        assert_allclose(saved.header["dim"][:4], np.array([3, 384, 384, 19]))

    def test_inverse(self):
        loader = Compose([LoadImaged(("im1", "im2"), image_only=True), EnsureChannelFirstd(("im1", "im2"))])
        data = loader({"im1": self.fnames[0], "im2": self.fnames[1]})
        tr = ResampleToMatch()
        im_mod = tr(data["im2"], data["im1"])
        self.assertNotEqual(im_mod.shape, data["im2"].shape)
        self.assertGreater(((im_mod.affine - data["im2"].affine) ** 2).sum() ** 0.5, 1e-2)
        # inverse
        im_mod2 = tr.inverse(im_mod)
        self.assertEqual(im_mod2.shape, data["im2"].shape)
        self.assertLess(((im_mod2.affine - data["im2"].affine) ** 2).sum() ** 0.5, 1e-2)
        self.assertEqual(im_mod2.applied_operations, [])

    def test_no_name(self):
        img_1 = MetaTensor(torch.zeros(1, 2, 2, 2))
        img_2 = MetaTensor(torch.zeros(1, 3, 3, 3))
        im_mod = ResampleToMatch()(img_1, img_2)
        self.assertEqual(im_mod.meta["filename_or_obj"], "resample_to_match_source")
        SaveImage(output_dir=self.tmpdir, output_postfix="", separate_folder=False, resample=False)(im_mod)


if __name__ == "__main__":
    unittest.main()
