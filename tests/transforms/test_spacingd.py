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

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.transforms import Spacingd
from monai.utils import ensure_tuple_rep
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_DEVICES, assert_allclose, skip_if_quick

TESTS: list[tuple] = []
for device in TEST_DEVICES:
    TESTS.append(
        (
            "spacing 3d",
            {"image": MetaTensor(torch.ones((2, 10, 15, 20)), affine=torch.eye(4))},
            dict(keys="image", pixdim=(1, 2, 1.4)),
            (2, 10, 8, 15),
            torch.as_tensor(np.diag([1, 2, 1.4, 1.0])),
            *device,
        )
    )
    TESTS.append(
        (
            "spacing 2d",
            {"image": MetaTensor(torch.ones((2, 10, 20)), affine=torch.eye(3))},
            dict(keys="image", pixdim=(1, 2)),
            (2, 10, 10),
            torch.as_tensor(np.diag((1, 2, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "spacing 2d no metadata",
            {"image": MetaTensor(torch.ones((2, 10, 20)))},
            dict(keys="image", pixdim=(1, 2)),
            (2, 10, 10),
            torch.as_tensor(np.diag((1, 2, 1, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "interp all",
            {
                "image": MetaTensor(np.arange(20).reshape((2, 1, 10)), affine=torch.eye(4)),
                "seg": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
            },
            dict(keys=("image", "seg"), mode="nearest", pixdim=(1, 0.2)),
            (2, 1, 46),
            torch.as_tensor(np.diag((1, 0.2, 1, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "interp sep",
            {
                "image": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
                "seg": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
            },
            dict(keys=("image", "seg"), mode=("bilinear", "nearest"), pixdim=(1, 0.2)),
            (2, 1, 46),
            torch.as_tensor(np.diag((1, 0.2, 1, 1))),
            *device,
        )
    )
    TESTS.append(
        (
            "interp sep",
            {
                "image": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
                "seg1": MetaTensor(torch.ones((2, 1, 10)), affine=torch.diag(torch.tensor([2, 2, 2, 1]))),
                "seg2": MetaTensor(torch.ones((2, 1, 10)), affine=torch.eye(4)),
            },
            dict(keys=("image", "seg1", "seg2"), mode=("bilinear", "nearest", "nearest"), pixdim=(1, 1, 1)),
            (2, 1, 10),
            torch.as_tensor(np.diag((1, 1, 1, 1))),
            *device,
        )
    )

TESTS_TORCH = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TESTS_TORCH.append([{"keys": "seg", "pixdim": [0.2, 0.3, 1]}, torch.ones(2, 1, 2, 3), track_meta, *device])


class TestSpacingDCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_spacingd(self, _, data, kw_args, expected_shape, expected_affine, device):
        data = {k: v.to(device) for k, v in data.items()}
        tr = Spacingd(**kw_args)
        call_param = {"data": data}
        res = tr(**call_param)
        # test lazy
        if not isinstance(kw_args["keys"], str):  # multiple keys
            kw_args["mode"] = ensure_tuple_rep(kw_args["mode"], len(kw_args["keys"]))
            init_param = kw_args.copy()
            for key, mode in zip(kw_args["keys"], kw_args["mode"]):
                init_param["keys"], init_param["mode"] = key, mode
                test_resampler_lazy(tr, res, init_param, call_param, output_key=key)
        else:
            test_resampler_lazy(tr, res, kw_args, call_param, output_key=kw_args["keys"])
        in_img = data["image"]
        out_img = res["image"]
        self.assertEqual(in_img.device, out_img.device)
        # no change in number of keys
        self.assertEqual(tuple(sorted(data)), tuple(sorted(res)))
        np.testing.assert_allclose(out_img.shape, expected_shape)
        assert_allclose(out_img.affine, expected_affine)

    @parameterized.expand(TESTS_TORCH)
    def test_orntd_torch(self, init_param, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        tr = Spacingd(**init_param)
        call_param = {"data": {"seg": img.to(device)}}
        res_data = tr(**call_param)  # type: ignore
        res = res_data["seg"]

        if track_meta:
            test_resampler_lazy(tr, res_data, init_param, call_param, output_key="seg")
            self.assertIsInstance(res, MetaTensor)
            assert isinstance(res, MetaTensor)  # for mypy type narrowing
            new_spacing = affine_to_spacing(res.affine, 3)
            assert_allclose(new_spacing, init_param["pixdim"], type_test=False)
            self.assertNotEqual(img.shape, res.shape)
        else:
            self.assertIsInstance(res, torch.Tensor)
            self.assertNotIsInstance(res, MetaTensor)
            self.assertNotEqual(img.shape, res.shape)

    @skip_if_quick
    def test_space_same_shape(self):
        affine_1 = np.array(
            [
                [1.499277e00, 2.699563e-02, 3.805804e-02, -1.948635e02],
                [-2.685805e-02, 1.499757e00, -2.635604e-12, 4.438188e01],
                [-3.805194e-02, -5.999028e-04, 1.499517e00, 4.036536e01],
                [0.000000e00, 0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
        affine_2 = np.array(
            [
                [1.499275e00, 2.692252e-02, 3.805728e-02, -1.948635e02],
                [-2.693010e-02, 1.499758e00, -4.260525e-05, 4.438188e01],
                [-3.805190e-02, -6.406730e-04, 1.499517e00, 4.036536e01],
                [0.000000e00, 0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
        img_1 = MetaTensor(np.zeros((1, 238, 145, 315)), affine=affine_1)
        img_2 = MetaTensor(np.zeros((1, 238, 145, 315)), affine=affine_2)
        out = Spacingd(("img_1", "img_2"), pixdim=1)({"img_1": img_1, "img_2": img_2})
        self.assertEqual(out["img_1"].shape, out["img_2"].shape)  # ensure_same_shape True
        out = Spacingd(("img_1", "img_2"), pixdim=1, ensure_same_shape=False)({"img_1": img_1, "img_2": img_2})
        self.assertNotEqual(out["img_1"].shape, out["img_2"].shape)  # ensure_same_shape False


if __name__ == "__main__":
    unittest.main()
