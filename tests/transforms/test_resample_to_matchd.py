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

from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    Invertd,
    Lambda,
    LoadImaged,
    ResampleToMatchd,
    SaveImaged,
)
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import assert_allclose, download_url_or_skip_test, testing_data_config


def update_fname(d):
    d["im3"].meta["filename_or_obj"] = "file3.nii.gz"
    return d


class TestResampleToMatchd(unittest.TestCase):
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

    def test_correct(self):
        transforms = Compose(
            [
                LoadImaged(("im1", "im2")),
                EnsureChannelFirstd(("im1", "im2")),
                CopyItemsd(("im2"), names=("im3")),
                ResampleToMatchd("im3", "im1"),
                Lambda(update_fname),
                SaveImaged("im3", output_dir=self.tmpdir, output_postfix="", separate_folder=False, resample=False),
            ]
        )
        data = transforms({"im1": self.fnames[0], "im2": self.fnames[1]})
        # check that output sizes match
        assert_allclose(data["im1"].shape, data["im3"].shape)
        # and that the meta data has been updated accordingly
        assert_allclose(data["im3"].affine, data["im1"].affine)
        # check we're different from the original
        self.assertTrue(any(i != j for i, j in zip(data["im3"].shape, data["im2"].shape)))
        self.assertTrue(any(i != j for i, j in zip(data["im3"].affine.flatten(), data["im2"].affine.flatten())))
        # test the inverse
        data = Invertd("im3", transforms)(data)
        assert_allclose(data["im2"].shape, data["im3"].shape)

    def test_lazy(self):
        pre_transforms = Compose(
            [LoadImaged(("im1", "im2")), EnsureChannelFirstd(("im1", "im2")), CopyItemsd(("im2"), names=("im3"))]
        )
        data = pre_transforms({"im1": self.fnames[0], "im2": self.fnames[1]})
        init_param = {"keys": "im3", "key_dst": "im1"}
        resampler = ResampleToMatchd(**init_param)
        call_param = {"data": data}
        non_lazy_out = resampler(**call_param)
        test_resampler_lazy(resampler, non_lazy_out, init_param, call_param, output_key="im3")


if __name__ == "__main__":
    unittest.main()
