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

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import SegResNetDS, SegResNetDS2
from tests.util import SkipIfBeforePyTorchVersion, test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CASE_SEGRESNET_DS = []
for spatial_dims in range(2, 4):
    for init_filters in [8, 16]:
        for act in ["relu", "leakyrelu"]:
            for norm in ["BATCH", ("instance", {"affine": True})]:
                for upsample_mode in ["deconv", "nontrainable"]:
                    test_case = [
                        {
                            "spatial_dims": spatial_dims,
                            "init_filters": init_filters,
                            "act": act,
                            "norm": norm,
                            "upsample_mode": upsample_mode,
                        },
                        (2, 1, *([16] * spatial_dims)),
                        (2, 2, *([16] * spatial_dims)),
                    ]
                    TEST_CASE_SEGRESNET_DS.append(test_case)

TEST_CASE_SEGRESNET_DS2 = []
for spatial_dims in range(2, 4):
    for out_channels in [1, 2]:
        for dsdepth in [1, 2, 3]:
            test_case = [
                {"spatial_dims": spatial_dims, "init_filters": 8, "out_channels": out_channels, "dsdepth": dsdepth},
                (2, 1, *([16] * spatial_dims)),
                (2, out_channels, *([16] * spatial_dims)),
            ]
            TEST_CASE_SEGRESNET_DS2.append(test_case)

TEST_CASE_SEGRESNET_DS3 = [
    ({"init_filters": 8, "dsdepth": 2, "resolution": None}, (2, 1, 16, 16, 16), ((2, 2, 16, 16, 16), (2, 2, 8, 8, 8))),
    (
        {"init_filters": 8, "dsdepth": 3, "resolution": None},
        (2, 1, 16, 16, 16),
        ((2, 2, 16, 16, 16), (2, 2, 8, 8, 8), (2, 2, 4, 4, 4)),
    ),
    (
        {"init_filters": 8, "dsdepth": 3, "resolution": [1, 1, 5]},
        (2, 1, 16, 16, 16),
        ((2, 2, 16, 16, 16), (2, 2, 8, 8, 16), (2, 2, 4, 4, 16)),
    ),
    (
        {"init_filters": 8, "dsdepth": 3, "resolution": [1, 2, 5]},
        (2, 1, 16, 16, 16),
        ((2, 2, 16, 16, 16), (2, 2, 8, 8, 16), (2, 2, 4, 8, 16)),
    ),
]


class TestSegResNetDS(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SEGRESNET_DS)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SegResNetDS(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    @parameterized.expand(TEST_CASE_SEGRESNET_DS)
    def test_shape_ds2(self, input_param, input_shape, expected_shape):
        net = SegResNetDS2(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device), with_label=False)
            self.assertEqual(result[0].shape, expected_shape, msg=str(input_param))
            self.assertTrue(result[1] == [])

            result = net(torch.randn(input_shape).to(device), with_point=False)
            self.assertEqual(result[1].shape, expected_shape, msg=str(input_param))
            self.assertTrue(result[0] == [])

    @parameterized.expand(TEST_CASE_SEGRESNET_DS2)
    def test_shape2(self, input_param, input_shape, expected_shape):
        dsdepth = input_param.get("dsdepth", 1)
        for net in [SegResNetDS, SegResNetDS2]:
            net = net(**input_param).to(device)
            net.train()
            if isinstance(net, SegResNetDS2):
                result = net(torch.randn(input_shape).to(device), with_label=False)[0]
            else:
                result = net(torch.randn(input_shape).to(device))
            if dsdepth > 1:
                assert isinstance(result, list)
                self.assertEqual(dsdepth, len(result))
                for i in range(dsdepth):
                    self.assertEqual(
                        result[i].shape,
                        expected_shape[:2] + tuple(e // (2**i) for e in expected_shape[2:]),
                        msg=str(input_param),
                    )
            else:
                assert isinstance(result, torch.Tensor)
                self.assertEqual(result.shape, expected_shape, msg=str(input_param))

            if not isinstance(net, SegResNetDS2):
                # eval mode of SegResNetDS2 has same output as training mode
                # so only test eval mode for SegResNetDS
                net.eval()
                result = net(torch.randn(input_shape).to(device))
                assert isinstance(result, torch.Tensor)
                self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    @parameterized.expand(TEST_CASE_SEGRESNET_DS3)
    def test_shape3(self, input_param, input_shape, expected_shapes):
        dsdepth = input_param.get("dsdepth", 1)
        for net in [SegResNetDS, SegResNetDS2]:
            net = net(**input_param).to(device)
            net.train()
            if isinstance(net, SegResNetDS2):
                result = net(torch.randn(input_shape).to(device), with_point=False)[1]
            else:
                result = net(torch.randn(input_shape).to(device))
            assert isinstance(result, list)
            self.assertEqual(dsdepth, len(result))
            for i in range(dsdepth):
                self.assertEqual(result[i].shape, expected_shapes[i], msg=str(input_param))

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SegResNetDS(spatial_dims=4)

        with self.assertRaises(ValueError):
            SegResNetDS2(spatial_dims=4)

    @SkipIfBeforePyTorchVersion((1, 10))
    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_SEGRESNET_DS[0]
        net = SegResNetDS(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
