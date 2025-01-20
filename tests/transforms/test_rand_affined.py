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
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandAffined
from monai.utils import GridSampleMode, ensure_tuple_rep
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import assert_allclose, is_tf32_env

_rtol = 1e-3 if is_tf32_env() else 1e-4

TESTS = []

for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
    TESTS.append(
        [
            dict(device=device, spatial_size=None, keys=("img", "seg")),
            {
                "img": MetaTensor(torch.arange(27).reshape((3, 3, 3))),
                "seg": MetaTensor(torch.arange(27).reshape((3, 3, 3))),
            },
            torch.arange(27).reshape((3, 3, 3)),
        ]
    )
    TESTS.append(
        [
            dict(device=device, spatial_size=(2, 2), keys=("img", "seg")),
            {"img": MetaTensor(torch.ones((3, 3, 3))), "seg": MetaTensor(torch.ones((3, 3, 3)))},
            torch.ones((3, 2, 2)),
        ]
    )
    TESTS.append(
        [
            dict(device=device, spatial_size=(2, 2), cache_grid=True, keys=("img", "seg")),
            {"img": MetaTensor(torch.ones((3, 3, 3))), "seg": MetaTensor(torch.ones((3, 3, 3)))},
            torch.ones((3, 2, 2)),
        ]
    )
    TESTS.append(
        [
            dict(device=device, spatial_size=(2, 2, 2), keys=("img", "seg")),
            {"img": MetaTensor(torch.ones((1, 3, 3, 3))), "seg": MetaTensor(torch.ones((1, 3, 3, 3)))},
            torch.ones((1, 2, 2, 2)),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                spatial_size=(2, 2, 2),
                padding_mode="zeros",
                device=device,
                keys=("img", "seg"),
                mode="bilinear",
            ),
            {"img": MetaTensor(torch.ones((1, 3, 3, 3))), "seg": MetaTensor(torch.ones((1, 3, 3, 3)))},
            torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            torch.tensor([[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            {
                "img": MetaTensor(
                    torch.tensor(
                        [
                            [
                                [18.736153, 15.581954, 12.4277525],
                                [27.398798, 24.244598, 21.090399],
                                [36.061443, 32.90724, 29.753046],
                            ]
                        ]
                    )
                ),
                "seg": MetaTensor(torch.tensor([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]])),
            },
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                spatial_size=(2, 2, 2),
                padding_mode="zeros",
                device=device,
                keys=("img", "seg"),
                mode=GridSampleMode.BILINEAR,
            ),
            {"img": MetaTensor(torch.ones((1, 3, 3, 3))), "seg": MetaTensor(torch.ones((1, 3, 3, 3)))},
            torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            {
                "img": MetaTensor(
                    np.array(
                        [
                            [
                                [18.736153, 15.581954, 12.4277525],
                                [27.398798, 24.244598, 21.090399],
                                [36.061443, 32.90724, 29.753046],
                            ]
                        ]
                    )
                ),
                "seg": MetaTensor(np.array([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]])),
            },
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                cache_grid=True,
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            {
                "img": MetaTensor(
                    torch.tensor(
                        [
                            [
                                [18.736153, 15.581954, 12.4277525],
                                [27.398798, 24.244598, 21.090399],
                                [36.061443, 32.90724, 29.753046],
                            ]
                        ]
                    )
                ),
                "seg": MetaTensor(torch.tensor([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]])),
            },
        ]
    )


class TestRandAffined(unittest.TestCase):
    @parameterized.expand(x + [y] for x, y in itertools.product(TESTS, (False, True)))
    def test_rand_affined(self, input_param, input_data, expected_val, track_meta):
        set_track_meta(track_meta)
        g = RandAffined(**input_param).set_random_state(123)
        call_param = {"data": input_data}
        res = g(**call_param)
        # test lazy
        if track_meta and input_data["img"].ndim in (3, 4):
            if "mode" not in input_param.keys():
                input_param["mode"] = "bilinear"
            if not isinstance(input_param["keys"], str):
                input_param["mode"] = ensure_tuple_rep(input_param["mode"], len(input_param["keys"]))
            lazy_init_param = input_param.copy()
            for key, mode in zip(input_param["keys"], input_param["mode"]):
                lazy_init_param["keys"], lazy_init_param["mode"] = key, mode
                resampler = RandAffined(**lazy_init_param).set_random_state(123)
                expected_output = resampler(**call_param)
                test_resampler_lazy(
                    resampler, expected_output, lazy_init_param, call_param, seed=123, output_key=key, rtol=_rtol
                )
            resampler.lazy = False

        if input_param.get("cache_grid", False):
            self.assertIsNotNone(g.rand_affine._cached_grid)
        for key in res:
            if isinstance(key, str) and key.endswith("_transforms"):
                continue
            result = res[key]
            if track_meta:
                self.assertIsInstance(result, MetaTensor)
                self.assertEqual(len(result.applied_operations), 1)
            expected = expected_val[key] if isinstance(expected_val, dict) else expected_val
            assert_allclose(result, expected, rtol=_rtol, atol=1e-3, type_test=False)

        g.set_random_state(4)
        res = g(**call_param)
        if not track_meta:
            return

        # affine should be tensor because the resampler only supports pytorch backend
        if isinstance(res["img"], MetaTensor) and "extra_info" in res["img"].applied_operations[0]:
            if not res["img"].applied_operations[-1]["extra_info"]:
                return
            if not res["img"].applied_operations[-1]["extra_info"]["extra_info"]["do_resampling"]:
                return
            affine_img = res["img"].applied_operations[0]["extra_info"]["extra_info"]["affine"]
            affine_seg = res["seg"].applied_operations[0]["extra_info"]["extra_info"]["affine"]
            assert_allclose(affine_img, affine_seg, rtol=_rtol, atol=1e-3)

        res_inv = g.inverse(res)
        for k, v in res_inv.items():
            self.assertIsInstance(v, MetaTensor)
            self.assertEqual(len(v.applied_operations), 0)
            self.assertTupleEqual(v.shape, input_data[k].shape)

    @parameterized.expand([(None,), ((2, -1),)])  # spatial size is None  # spatial size is dynamic
    def test_ill_cache(self, spatial_size):
        with self.assertWarns(UserWarning):
            RandAffined(device=device, spatial_size=spatial_size, prob=1.0, cache_grid=True, keys=("img", "seg"))


if __name__ == "__main__":
    unittest.main()
